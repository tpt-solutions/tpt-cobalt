import os, tomllib, re, glob

ROOT = r"D:\Programming\1PRODUCTION\Open Source\tpt-cobalt"
FORKED = os.path.join(ROOT, "forked")

def quote(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[%s]" % ", ".join(quote(x) for x in v)
    if isinstance(v, dict):
        return ws_table_inline(v)
    return '"%s"' % str(v).replace("\\", "\\\\").replace('"', '\\"')

def ws_table_inline(d):
    return "{ " + ", ".join("%s = %s" % (k, quote(v)) for k, v in d.items()) + " }"

for repo in sorted(d for d in os.listdir(FORKED) if os.path.isdir(os.path.join(FORKED, d))):
    orig = os.path.join(FORKED, repo, "Cargo.toml.orig")
    if not os.path.exists(orig):
        continue
    with open(orig, "rb") as f:
        rdata = tomllib.load(f)
    ws = rdata.get("workspace", {})
    pkg = ws.get("package", {})
    wdeps = ws.get("dependencies", {})
    wlints = ws.get("lints", {})
    repo_root = os.path.join(FORKED, repo)

    # precompute internal target dirs from workspace.dependencies paths
    def resolve_target(name):
        base = wdeps.get(name)
        if isinstance(base, dict) and "path" in base:
            return os.path.normpath(os.path.join(repo_root, base["path"]))
        return None

    def concrete_spec(name, crate_dir, overrides):
        base = wdeps.get(name)
        parts = []
        if isinstance(base, str):
            if "/" in base or base.startswith("."):
                tgt = os.path.normpath(os.path.join(repo_root, base))
                rel = os.path.relpath(tgt, crate_dir).replace("\\", "/")
                parts.append('path = "%s"' % rel)
            else:
                parts.append('version = "%s"' % base)
        elif isinstance(base, dict):
            tgt = resolve_target(name)
            if tgt is not None:
                rel = os.path.relpath(tgt, crate_dir).replace("\\", "/")
                parts.append('path = "%s"' % rel)
            if "version" in base:
                parts.append('version = "%s"' % base["version"])
            if "default-features" in base:
                parts.append("default-features = %s" % str(base["default-features"]).lower())
            if "features" in base:
                parts.append("features = [%s]" % ", ".join('"%s"' % x for x in base["features"]))
            if "optional" in base:
                parts.append("optional = %s" % str(base["optional"]).lower())
            if "git" in base:
                parts.append('git = "%s"' % base["git"])
        # apply crate-level overrides
        if "default-features" in overrides:
            # replace any earlier default-features
            parts = [p for p in parts if not p.startswith("default-features")]
            parts.append("default-features = %s" % str(overrides["default-features"]).lower())
        if "features" in overrides:
            parts = [p for p in parts if not p.startswith("features")]
            parts.append("features = [%s]" % ", ".join('"%s"' % x for x in overrides["features"]))
        if "optional" in overrides:
            parts = [p for p in parts if not p.startswith("optional")]
            parts.append("optional = %s" % str(overrides["optional"]).lower())
        if "package" in overrides:
            parts.append('package = "%s"' % overrides["package"])
        return "{ " + ", ".join(parts) + " }"

    for cf in glob.glob(os.path.join(repo_root, "crates", "**", "Cargo.toml"), recursive=True):
        crate_dir = os.path.dirname(cf)
        txt = open(cf, encoding="utf-8").read()

        # (a) [lints] workspace = true
        def lint_repl(m):
            if wlints:
                body = "\n".join("%s = %s" % (k, ws_table_inline(v) if isinstance(v, dict) else quote(v))
                                 for k, v in wlints.items())
                return "[lints]\n" + body
            return "[lints]"
        txt = re.sub(r'(?ms)\[lints\]\s*\n\s*workspace\s*=\s*true', lint_repl, txt)

        # (b) package-ish fields: key.workspace = true  (version/edition/license/rust-version/authors/...)
        txt = re.sub(r'(?m)^(\s*)(edition|version|license|rust-version|authors|repository|homepage|documentation|description|readme)\.workspace\s*=\s*true',
                     lambda m: "%s%s = %s" % (m.group(1), m.group(2), quote(pkg.get(m.group(2), ""))), txt)

        # (c) dependency dot-form: name.workspace = true
        def dep_dot_repl(m):
            name = m.group(2)
            return "%s%s = %s" % (m.group(1), name, concrete_spec(name, crate_dir, {}))
        txt = re.sub(r'(?m)^(\s*)([a-zA-Z][a-zA-Z0-9\-]*)\.workspace\s*=\s*true',
                     dep_dot_repl, txt)

        # (d) dependency inline-form: name = { workspace = true, overrides }
        def dep_inline_repl(m):
            name = m.group(2)
            body = m.group(3)
            overrides = {}
            om = re.search(r'default-features\s*=\s*(true|false)', body)
            if om:
                overrides["default-features"] = (om.group(1) == "true")
            fm = re.search(r'features\s*=\s*\[([^\]]*)\]', body)
            if fm:
                overrides["features"] = [x.strip().strip('"') for x in fm.group(1).split(",") if x.strip()]
            om2 = re.search(r'optional\s*=\s*(true|false)', body)
            if om2:
                overrides["optional"] = (om2.group(1) == "true")
            pm = re.search(r'package\s*=\s*"([^"]+)"', body)
            if pm:
                overrides["package"] = pm.group(1)
            return "%s = %s" % (name, concrete_spec(name, crate_dir, overrides))
        txt = re.sub(r'(?m)^(\s*)([a-zA-Z][a-zA-Z0-9\-]*)\s*=\s*\{\s*workspace\s*=\s*true([^}]*)\}',
                     dep_inline_repl, txt)

        open(cf, "w", encoding="utf-8").write(txt)

print("CONCRETIZE DONE")
