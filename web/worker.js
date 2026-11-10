// Author: Mohammad Nour
// Runs our Python code (vrp_project/src) inside the browser with Pyodide.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

// __init__.py is empty, and GitHub Pages hides files that start with "_", so we create it here
const SRC_FILES = ["problem.py", "ga.py", "pso.py", "aco.py",
                   "dp_baseline.py", "experiment.py"];
let demo = null;

async function getText(path) {
  const res = await fetch(new URL(path, self.location.href));
  if (!res.ok) throw new Error("cannot load " + path + " (" + res.status + ")");
  return res.text();
}

async function init() {
  postMessage({ type: "status", key: "python" });
  const py = await loadPyodide();
  postMessage({ type: "status", key: "numpy" });
  await py.loadPackage("numpy");
  postMessage({ type: "status", key: "code" });
  py.FS.mkdirTree("/home/pyodide/src");
  py.FS.writeFile("/home/pyodide/src/__init__.py", "");
  for (const f of SRC_FILES) {
    py.FS.writeFile("/home/pyodide/src/" + f, await getText("../vrp_project/src/" + f));
  }
  py.FS.writeFile("/home/pyodide/demo.py", await getText("demo.py"));
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  demo = py.pyimport("demo");
  postMessage({ type: "ready" });
}

const ready = init().catch((e) => postMessage({ type: "fatal", text: String(e) }));

onmessage = async (e) => {
  await ready;
  const { id, fn, args } = e.data;
  try {
    const out = demo[fn](...args);
    postMessage({ type: "result", id, data: JSON.parse(out) });
  } catch (err) {
    postMessage({ type: "error", id, text: String(err) });
  }
};
