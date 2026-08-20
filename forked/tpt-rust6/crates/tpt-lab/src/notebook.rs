//! The reactive [`Notebook`]: named cells, a dependency DAG, and incremental
//! re-execution of exactly the transitively dependent cells.

use std::any::{type_name, Any};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::render::{render_value, Render};
use crate::value::{literal_repr, Value};
use crate::LabError;

/// Re-evaluates a derived cell against the current notebook state.
pub type Compute = Arc<dyn Fn(&Notebook) -> Result<Value, LabError> + Send + Sync>;
/// Renders a cell value (installed by `Notebook::set_rendered`).
pub type Renderer = Arc<dyn Fn(&Value, &Notebook) -> String + Send + Sync>;

/// One notebook cell: its current value, its declared dependencies, the source
/// expression used for export, and (for derived cells) its closure.
#[derive(Clone)]
pub struct Cell {
    value: Value,
    deps: Vec<String>,
    expr: String,
    compute: Option<Compute>,
    render: Option<Renderer>,
    error: Option<String>,
    seq: u64,
}

impl Cell {
    /// The cell's current value.
    pub fn value(&self) -> &Value {
        &self.value
    }
    /// Recorded static type name of the value.
    pub fn type_name(&self) -> &'static str {
        self.value.type_name()
    }
    /// Names of the cells this cell depends on.
    pub fn deps(&self) -> &[String] {
        &self.deps
    }
    /// Source expression (or literal) recorded for `export_rust`.
    pub fn expr(&self) -> &str {
        &self.expr
    }
    /// `true` if the cell has a closure and is re-executed reactively.
    pub fn is_derived(&self) -> bool {
        self.compute.is_some()
    }
    /// The error from the most recent failed re-execution, if any. The cell
    /// keeps its previous (now stale) value in that case.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

/// A reactive notebook of named, type-tracked cells.
///
/// ```
/// use tpt_lab::prelude::*;
///
/// let mut nb = Notebook::new();
/// nb.set("A", 1i64).unwrap();
/// nb.set_expr("B", "A + 1", &["A"], |nb: &Notebook| Ok(nb.try_get::<i64>("A")? + 1)).unwrap();
/// assert_eq!(nb.get::<i64>("B"), Some(&2));
///
/// nb.set("A", 10i64).unwrap(); // B is re-executed automatically
/// assert_eq!(nb.get::<i64>("B"), Some(&11));
/// ```
#[derive(Default)]
pub struct Notebook {
    cells: BTreeMap<String, Cell>,
    seq: u64,
}

impl Notebook {
    /// An empty notebook.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }
    /// `true` if the notebook has no cells.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
    /// `true` if `name` is defined.
    pub fn contains(&self, name: &str) -> bool {
        self.cells.contains_key(name)
    }
    /// Borrow a cell.
    pub fn cell(&self, name: &str) -> Option<&Cell> {
        self.cells.get(name)
    }
    /// Cell names in definition order.
    pub fn names(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.cells.keys().map(|s| s.as_str()).collect();
        out.sort_by_key(|n| self.cells[*n].seq);
        out
    }

    // ---------------------------------------------------------------- values

    /// Define or update a constant cell. The value's type must match the
    /// existing cell's type; use [`Notebook::redefine`] to change it.
    /// Dependent cells are re-executed immediately.
    pub fn set<T: Any + Send + Sync>(&mut self, name: &str, value: T) -> Result<(), LabError> {
        self.install(name, Value::new(value), Vec::new(), None, None, None, false)
    }

    /// Like [`Notebook::set`], but allows the cell's type to change. Dependent
    /// cells are re-executed and may fail with [`LabError::TypeMismatch`].
    pub fn redefine<T: Any + Send + Sync>(&mut self, name: &str, value: T) -> Result<(), LabError> {
        self.install(name, Value::new(value), Vec::new(), None, None, None, true)
    }

    /// Define a constant cell together with a custom [`Render`] implementation.
    pub fn set_rendered<T: Render + Any + Send + Sync>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<(), LabError> {
        let renderer: Renderer = Arc::new(|v: &Value, nb: &Notebook| match v.downcast_ref::<T>() {
            Some(v) => v.render(nb),
            None => format!("<{}>", v.type_name()),
        });
        self.install(
            name,
            Value::new(value),
            Vec::new(),
            None,
            None,
            Some(renderer),
            false,
        )
    }

    /// Define a derived cell.
    ///
    /// * `expr` is the source text recorded for [`Notebook::export_rust`].
    /// * `deps` are the cell names this cell reads; every one must already
    ///   exist, otherwise [`LabError::MissingDependency`] is returned.
    /// * `f` is re-run whenever any transitive dependency changes.
    ///
    /// The closure is evaluated once, immediately, so the cell always holds a
    /// real value.
    pub fn set_expr<T, F>(
        &mut self,
        name: &str,
        expr: &str,
        deps: &[&str],
        f: F,
    ) -> Result<(), LabError>
    where
        T: Any + Send + Sync,
        F: Fn(&Notebook) -> Result<T, LabError> + Send + Sync + 'static,
    {
        for d in deps {
            if !self.cells.contains_key(*d) {
                return Err(LabError::MissingDependency {
                    cell: name.to_string(),
                    dep: (*d).to_string(),
                });
            }
        }
        let compute: Compute = Arc::new(move |nb: &Notebook| f(nb).map(Value::new));
        let value = compute(self)?;
        self.install(
            name,
            value,
            deps.iter().map(|d| d.to_string()).collect(),
            Some(expr.to_string()),
            Some(compute),
            None,
            false,
        )
    }

    /// Register extra DAG edges for an existing cell, then re-execute it and
    /// everything downstream. Cycles are rejected with [`LabError::Cycle`] and
    /// leave the graph unchanged.
    pub fn depends_on(&mut self, name: &str, deps: &[&str]) -> Result<(), LabError> {
        if !self.cells.contains_key(name) {
            return Err(LabError::Undefined(name.to_string()));
        }
        for d in deps {
            if !self.cells.contains_key(*d) {
                return Err(LabError::MissingDependency {
                    cell: name.to_string(),
                    dep: (*d).to_string(),
                });
            }
        }
        let previous = self.cells[name].deps.clone();
        {
            let cell = self.cells.get_mut(name).unwrap();
            for d in deps {
                if !cell.deps.iter().any(|x| x == *d) {
                    cell.deps.push((*d).to_string());
                }
            }
        }
        if let Err(e) = self.topo_order() {
            self.cells.get_mut(name).unwrap().deps = previous;
            return Err(e);
        }
        self.eval_cell(name)?;
        self.recompute_dependents(name)
    }

    /// Remove a cell. Cells that depended on it keep their last value but will
    /// fail on the next re-execution; the dangling edge is reported by
    /// [`Notebook::check`].
    pub fn remove(&mut self, name: &str) -> Option<Cell> {
        self.cells.remove(name)
    }

    // ----------------------------------------------------------------- reads

    /// Borrow a cell's value as `T`, or `None` if undefined or mistyped.
    pub fn get<T: Any>(&self, name: &str) -> Option<&T> {
        self.cells.get(name)?.value.downcast_ref::<T>()
    }

    /// Borrow a cell's value as `T`, reporting *why* it failed. This is the
    /// accessor derived cells should use inside their closures, so that a
    /// missing or retyped upstream cell surfaces as an error instead of a panic.
    pub fn try_get<T: Any>(&self, name: &str) -> Result<&T, LabError> {
        let cell = self
            .cells
            .get(name)
            .ok_or_else(|| LabError::Undefined(name.to_string()))?;
        cell.value
            .downcast_ref::<T>()
            .ok_or_else(|| LabError::TypeMismatch {
                cell: name.to_string(),
                expected: type_name::<T>().to_string(),
                actual: cell.value.type_name().to_string(),
            })
    }

    /// Runtime type registry lookup: the recorded type name of a cell.
    ///
    /// Note: type checking in tpt-lab is *runtime* (by `TypeId` / type name).
    /// Compile-time cross-cell checking is future work.
    pub fn type_of(&self, name: &str) -> Option<&str> {
        self.cells.get(name).map(|c| c.value.type_name())
    }

    /// Direct dependencies of a cell.
    pub fn dependencies(&self, name: &str) -> Vec<&str> {
        self.cells
            .get(name)
            .map(|c| c.deps.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Direct dependents of a cell.
    pub fn dependents(&self, name: &str) -> Vec<&str> {
        let mut out: Vec<&str> = self
            .cells
            .iter()
            .filter(|(_, c)| c.deps.iter().any(|d| d == name))
            .map(|(n, _)| n.as_str())
            .collect();
        out.sort_by_key(|n| self.cells[*n].seq);
        out
    }

    /// Validate the DAG: reports dependencies pointing at undefined cells.
    pub fn check(&self) -> Result<(), LabError> {
        for (name, cell) in &self.cells {
            for d in &cell.deps {
                if !self.cells.contains_key(d) {
                    return Err(LabError::MissingDependency {
                        cell: name.clone(),
                        dep: d.clone(),
                    });
                }
            }
        }
        self.topo_order().map(|_| ())
    }

    // ------------------------------------------------------------- rendering

    /// Render the named cell's value (Markdown for tables, a text grid or SVG
    /// for tensors, `Display` for scalars).
    pub fn display(&self, name: &str) -> Result<String, LabError> {
        let cell = self
            .cells
            .get(name)
            .ok_or_else(|| LabError::Undefined(name.to_string()))?;
        Ok(match &cell.render {
            Some(r) => r(&cell.value, self),
            None => render_value(&cell.value, self),
        })
    }

    /// Render every cell as `name: <rendered value>`, in definition order.
    pub fn display_all(&self) -> String {
        let mut out = String::new();
        for name in self.names() {
            let body = self.display(name).unwrap_or_default();
            out.push_str(&format!("{name}: {body}\n"));
        }
        out
    }

    // --------------------------------------------------------------- export

    /// Generate a best-effort single-file Rust reproduction of the notebook.
    ///
    /// Cells are emitted in topological order. Derived cells emit the `expr`
    /// text they were registered with; constant cells emit a literal when one
    /// can be produced, otherwise a comment marking the value as opaque.
    pub fn export_rust(&self) -> String {
        let order = self.topo_order().unwrap_or_else(|_| {
            self.names()
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        });
        let mut out = String::from(
            "// Auto-generated by tpt-lab `Notebook::export_rust`.\n\
             // Best-effort reproduction of the notebook's cell assignments.\n\
             #![allow(unused_variables, non_snake_case)]\n\n\
             fn main() {\n",
        );
        for name in order {
            let cell = &self.cells[&name];
            if !cell.deps.is_empty() {
                out.push_str(&format!(
                    "    // cell `{}` depends on: {}\n",
                    name,
                    cell.deps.join(", ")
                ));
            }
            let ty = cell.value.type_name();
            if cell.expr.is_empty() {
                out.push_str(&format!(
                    "    // let {}: {} = <opaque value, not expressible as a literal>;\n",
                    ident(&name),
                    ty
                ));
            } else {
                out.push_str(&format!(
                    "    let {}: {} = {};\n",
                    ident(&name),
                    ty,
                    cell.expr
                ));
            }
        }
        out.push_str("}\n");
        out
    }

    // -------------------------------------------------------------- internals

    #[allow(clippy::too_many_arguments)]
    fn install(
        &mut self,
        name: &str,
        value: Value,
        deps: Vec<String>,
        expr: Option<String>,
        compute: Option<Compute>,
        render: Option<Renderer>,
        allow_type_change: bool,
    ) -> Result<(), LabError> {
        for d in &deps {
            if !self.cells.contains_key(d) {
                return Err(LabError::MissingDependency {
                    cell: name.to_string(),
                    dep: d.clone(),
                });
            }
        }
        let previous = self.cells.get(name).cloned();
        if let Some(prev) = &previous {
            if !allow_type_change && prev.value.type_name() != value.type_name() {
                return Err(LabError::TypeChanged {
                    cell: name.to_string(),
                    was: prev.value.type_name().to_string(),
                    now: value.type_name().to_string(),
                });
            }
        }
        let seq = match &previous {
            Some(p) => p.seq,
            None => {
                self.seq += 1;
                self.seq
            }
        };
        let expr = expr.unwrap_or_else(|| literal_repr(&value).unwrap_or_default());
        let render = render.or_else(|| previous.as_ref().and_then(|p| p.render.clone()));
        self.cells.insert(
            name.to_string(),
            Cell {
                value,
                deps,
                expr,
                compute,
                render,
                error: None,
                seq,
            },
        );
        if let Err(e) = self.topo_order() {
            match previous {
                Some(p) => {
                    self.cells.insert(name.to_string(), p);
                }
                None => {
                    self.cells.remove(name);
                }
            }
            return Err(e);
        }
        self.recompute_dependents(name)
    }

    /// Re-execute exactly the transitive dependents of `name`, in topological
    /// order. Cells outside that set are never touched.
    fn recompute_dependents(&mut self, name: &str) -> Result<(), LabError> {
        let affected = self.transitive_dependents(name);
        if affected.is_empty() {
            return Ok(());
        }
        for n in self.topo_order()? {
            if affected.contains(&n) {
                self.eval_cell(&n)?;
            }
        }
        Ok(())
    }

    fn eval_cell(&mut self, name: &str) -> Result<(), LabError> {
        let compute = match self.cells.get(name).and_then(|c| c.compute.clone()) {
            Some(c) => c,
            None => return Ok(()), // constant cell: nothing to re-run
        };
        match compute(self) {
            Ok(value) => {
                let cell = self.cells.get_mut(name).unwrap();
                if cell.value.type_name() != value.type_name() {
                    return Err(LabError::TypeChanged {
                        cell: name.to_string(),
                        was: cell.value.type_name().to_string(),
                        now: value.type_name().to_string(),
                    });
                }
                cell.value = value;
                cell.error = None;
                Ok(())
            }
            Err(e) => {
                if let Some(cell) = self.cells.get_mut(name) {
                    cell.error = Some(e.to_string());
                }
                Err(e)
            }
        }
    }

    fn transitive_dependents(&self, name: &str) -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![name.to_string()];
        while let Some(cur) = stack.pop() {
            for (n, cell) in &self.cells {
                if cell.deps.contains(&cur) && seen.insert(n.clone()) {
                    stack.push(n.clone());
                }
            }
        }
        seen
    }

    /// Kahn-style topological order over the whole DAG (definition order breaks
    /// ties, so the result is deterministic).
    fn topo_order(&self) -> Result<Vec<String>, LabError> {
        let mut remaining: Vec<&str> = self.names();
        let mut done: BTreeSet<&str> = BTreeSet::new();
        let mut out: Vec<String> = Vec::with_capacity(remaining.len());
        while !remaining.is_empty() {
            let mut progressed = false;
            let mut next = Vec::new();
            for n in remaining {
                let ready = self.cells[n]
                    .deps
                    .iter()
                    // an unknown dep can't be scheduled; `check` reports it
                    .all(|d| done.contains(d.as_str()) || !self.cells.contains_key(d));
                if ready {
                    done.insert(n);
                    out.push(n.to_string());
                    progressed = true;
                } else {
                    next.push(n);
                }
            }
            remaining = next;
            if !progressed {
                let mut cycle: Vec<&str> = remaining;
                cycle.sort_unstable();
                return Err(LabError::Cycle(cycle.join(" -> ")));
            }
        }
        Ok(out)
    }
}

/// Turn a cell name into a valid Rust identifier for the exporter.
fn ident(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ident_sanitizes() {
        assert_eq!(ident("my cell-1"), "my_cell_1");
        assert_eq!(ident("2x"), "_2x");
    }

    #[test]
    fn cycles_are_rejected_and_graph_restored() {
        let mut nb = Notebook::new();
        nb.set("a", 1i64).unwrap();
        nb.set_expr("b", "a", &["a"], |nb: &Notebook| {
            Ok(*nb.try_get::<i64>("a")?)
        })
        .unwrap();
        let err = nb.depends_on("a", &["b"]).unwrap_err();
        assert!(matches!(err, LabError::Cycle(_)));
        assert!(nb.dependencies("a").is_empty());
        assert!(nb.check().is_ok());
    }
}
