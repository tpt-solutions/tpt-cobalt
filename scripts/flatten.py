import os, tomllib, glob, re, shutil

ROOT = r"D:\Programming\1PRODUCTION\Open Source\tpt-cobalt"
FORKED = os.path.join(ROOT, "forked")

# ---------- load current top-level ----------
with open(os.path.join(ROOT, "Cargo.toml"), "rb") as f:
    top = tomllib.load(f)
ws = top["workspace"]
my_pkg = dict(ws.get("package", {}))

# Seed the glue crates + common third-party deps explicitly (don't rely on the
# on-disk manifest, which may be a stale generated copy).
my_deps = {
    "tpt-tensor": {"path": "crates/tpt-tensor"},
    "tpt-autograd": {"path": "crates/tpt-autograd"},
    "tpt-ml": {"path": "crates/tpt-ml"},
    "tpt-hub": {"path": "crates/tpt-hub"},
    "tpt-runtime": {"path": "crates/tpt-runtime"},
    "serde": {"version": "1", "features": ["derive"]},
    "serde_json": "1",
    "anyhow": "1",
    "thiserror": "2",
    "tracing": "0.1",
}

merged_pkg = dict(my_pkg)
merged_deps = dict(my_deps)
merged_lints = {}
profiles = {}
crate_members = set()

repos = sorted(d for d in os.listdir(FORKED) if os.path.isdir(os.path.join(FORKED, d)))

# ---------- gather from each forked root ----------
for repo in repos:
    p = os.path.join(FORKED, repo, "Cargo.toml")
    if not os.path.exists(p):
        continue
    with open(p, "rb") as f:
        data = tomllib.load(f)
    w = data.get("workspace", {})
    for k, v in w.get("package", {}).items():
        merged_pkg.setdefault(k, v)
    for k, v in w.get("dependencies", {}).items():
        merged_deps.setdefault(k, v)
    if "lints" in w:
        for k, v in w["lints"].items():
            merged_lints.setdefault(k, v)
    for m in w.get("members", []):
        crate_members.add("/".join(["forked", repo, m]))
    for k, v in data.get("profile", {}).items():
        profiles.setdefault(k, v)

# map crate name -> member path (relative to workspace root, forward slashes)
name_to_member = {}
for m in crate_members:
    name_to_member[m.split("/")[-1]] = m

# register glue crates (crates/*) so their workspace.dependencies entries survive
for cf in glob.glob(os.path.join(ROOT, "crates", "*", "Cargo.toml")):
    name = tomllib.load(open(cf, "rb")).get("package", {}).get("name")
    rel = os.path.relpath(os.path.dirname(cf), ROOT).replace("\\", "/")
    if name:
        name_to_member[name] = rel

# rewrite path entries in merged_deps to point at the correct forked crate
for name, v in list(merged_deps.items()):
    if isinstance(v, dict) and "path" in v and name in name_to_member:
        v["path"] = name_to_member[name]

# ensure standard keys win
for k in ("version", "edition", "license", "rust-version"):
    if k in my_pkg:
        merged_pkg[k] = my_pkg[k]

# ---------- repoint cross-repo path deps in crate manifests ----------
repo_names = set(repos)
pat = re.compile(r'path\s*=\s*"(\.\./(' + "|".join(repo_names) + r')/[^"]*)"')
for repo in repos:
    for cf in glob.glob(os.path.join(FORKED, repo, "crates", "**", "Cargo.toml"), recursive=True):
        txt = open(cf, encoding="utf-8").read()
        def repl(m):
            return 'path = "../../%s/%s"' % (m.group(2), m.group(1).split("/", 1)[1])
        new = pat.sub(repl, txt)
        if new != txt:
            open(cf, "w", encoding="utf-8").write(new)

# ---------- diagnostics: scan all crate manifests for workspace refs ----------
missing_pkg = set()
missing_dep = set()
uses_lints = False
for cf in glob.glob(os.path.join(FORKED, "**", "Cargo.toml"), recursive=True):
    txt = open(cf, encoding="utf-8").read()
    for m in re.finditer(r'^\s*([a-z\-]+)\.workspace\s*=\s*true', txt, re.M):
        key = m.group(1)
        if key in ("version", "edition", "license", "rust-version", "authors",
                  "repository", "homepage", "documentation", "description", "readme"):
            if key not in merged_pkg:
                missing_pkg.add(key)
        elif key == "lints":
            uses_lints = True
        else:
            missing_dep.add(key)
    if re.search(r'^\s*lints\s*=\s*true', txt, re.M) or 'workspace = true' in txt and 'lints' in txt:
        uses_lints = True

if uses_lints and not merged_lints:
    merged_lints = {"rust": {"warnings": "deny", "unexpected_cfgs": "warn"}}

# ---------- emit top-level Cargo.toml ----------
def fmt(v, indent=0):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return '"%s"' % v.replace("\\", "\\\\").replace('"', '\\"')
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        inner = ", ".join("%s = %s" % (k, fmt(val, indent + 1)) for k, val in v.items())
        return "{ %s }" % inner
    return str(v)

lines = []
lines.append("[workspace]")
lines.append('resolver = "2"')
lines.append("members = [")
for m in ["crates/*"] + sorted(crate_members):
    lines.append('    "%s",' % m)
lines.append("]")
lines.append("")
lines.append("[workspace.package]")
for k, v in merged_pkg.items():
    lines.append("%s = %s" % (k, fmt(v)))
lines.append("")
lines.append("[workspace.dependencies]")
for k, v in merged_deps.items():
    lines.append("%s = %s" % (k, fmt(v)))
lines.append("")
if merged_lints:
    lines.append("[workspace.lints]")
    for k, v in merged_lints.items():
        lines.append("%s = %s" % (k, fmt(v)))
    lines.append("")
for pk, pv in profiles.items():
    lines.append("[profile.%s]" % pk)
    for k, v in pv.items():
        lines.append("%s = %s" % (k, fmt(v)))
    lines.append("")

with open(os.path.join(ROOT, "Cargo.toml"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ---------- neutralize forked root manifests ----------
for repo in repos:
    p = os.path.join(FORKED, repo, "Cargo.toml")
    if os.path.exists(p):
        shutil.move(p, p + ".orig")
    lock = os.path.join(FORKED, repo, "Cargo.lock")
    if os.path.exists(lock):
        shutil.move(lock, lock + ".orig")

print("MEMBERS:", len(crate_members) + 1, "repos:", repos)
print("MISSING_PKG_KEYS:", sorted(missing_pkg))
print("MISSING_DEP_KEYS:", sorted(missing_dep))
print("USES_LINTS:", uses_lints, "MERGED_LINTS:", bool(merged_lints))
print("PROFILES:", list(profiles.keys()))
print("DONE")
