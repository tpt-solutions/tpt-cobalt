// tpt-physics WebGL playground frontend.
//
// Loads the wasm module built from `tpt-physics-wasm`, drives either the DEM
// or CFD solver one sub-step per frame, and renders the result with WebGL.

import init, { DemSimulation, CfdSimulation, SphSimulation } from "./tpt_physics_wasm.js";

// ---------------------------------------------------------------------------
// Scene builders (return the JSON string the wasm constructors expect)
// ---------------------------------------------------------------------------

function demDamBreak() {
  // A block of granular material that collapses under gravity (the classic
  // "dam break" free-surface granular flow).
  const particles = [];
  const r = 0.08;
  const rho = 2600;
  const nx = 14, ny = 22, nz = 8;
  const gap = 2.05 * r;
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++) {
        particles.push({
          position: [-1.2 + i * gap, r + j * gap, -0.6 + k * gap],
          velocity: [0, 0, 0],
          radius: r,
          density: rho,
        });
      }
  return JSON.stringify({
    dt: 2e-4,
    e_star: 1e8,
    friction: 0.4,
    restitution: 0.1,
    floor_y: 0.0,
    max_speed: 6.0,
    drag: 0.0,
    particles,
    obstacles: [{ kind: "plane", point: [0, 0, 0], normal: [0, 1, 0] }],
  });
}

function demColumn() {
  // A tall slender column that slumps outward — exercises the contact solver
  // and floor rest.
  const particles = [];
  const r = 0.1;
  const rho = 1500;
  const n = 26;
  const gap = 2.05 * r;
  for (let j = 0; j < n; j++)
    for (let a = 0; a < 6; a++) {
      const ang = (a / 6) * Math.PI * 2;
      particles.push({
        position: [Math.cos(ang) * 0.3, r + j * gap, Math.sin(ang) * 0.3],
        velocity: [0, 0, 0],
        radius: r,
        density: rho,
      });
    }
  return JSON.stringify({
    dt: 2e-4,
    e_star: 5e7,
    friction: 0.5,
    restitution: 0.05,
    floor_y: 0.0,
    max_speed: 5.0,
    drag: 1.0,
    particles,
    obstacles: [{ kind: "plane", point: [0, 0, 0], normal: [0, 1, 0] }],
  });
}

function sphDamBreak() {
  // A column of weakly-compressible fluid (SPH) that collapses under gravity —
  // the free-surface case the LBM mesh can't represent.
  return JSON.stringify({
    h: 0.04,
    rho0: 1000.0,
    c: 20.0,
    gamma: 1.0,
    mu: 0.5,
    gravity: [0, -9.81],
    domain: [1.0, 1.0],
    dt: 1e-4,
    block: { nx: 40, ny: 70, spacing: 0.025, origin: [0.02, 0.02] },
  });
}

function cfdCavity() {
  return JSON.stringify({
    nx: 200,
    ny: 120,
    tau: 0.53,
    x_boundary: "periodic",
    walls: "box",
    moving_lid: { row: 119, v: 0.08 },
    rho0: 1.0,
    u0: [0, 0],
  });
}

function cfdCylinder() {
  return JSON.stringify({
    nx: 240,
    ny: 100,
    tau: 0.6,
    x_boundary: "inlet",
    inlet_velocity: 0.08,
    walls: "horizontal",
    circles: [{ cx: 70, cy: 50, r: 12 }],
    force: [0, 0],
    rho0: 1.0,
    u0: [0, 0],
  });
}

function demCfdCoupledDem() {
  // A granular dam break — the DEM phase of the combined CFD-DEM demo.
  const particles = [];
  const r = 0.08;
  const rho = 2600;
  const nx = 14, ny = 22, nz = 8;
  const gap = 2.05 * r;
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++) {
        particles.push({
          position: [-1.2 + i * gap, r + j * gap, -0.6 + k * gap],
          velocity: [0, 0, 0],
          radius: r,
          density: rho,
        });
      }
  return JSON.stringify({
    dt: 2e-4,
    e_star: 1e8,
    friction: 0.4,
    restitution: 0.1,
    floor_y: 0.0,
    max_speed: 6.0,
    drag: 1.0,
    particles,
    obstacles: [{ kind: "plane", point: [0, 0, 0], normal: [0, 1, 0] }],
  });
}

function demCfdCoupledCfd() {
  // A lid-driven cavity — the CFD phase of the combined CFD-DEM demo. The two
  // run side-by-side in one render loop (one-way demo; two-way back-coupling
  // lives in the Rust example `coupled_dem_cfd.rs`).
  return JSON.stringify({
    nx: 160,
    ny: 100,
    tau: 0.53,
    x_boundary: "periodic",
    walls: "box",
    moving_lid: { row: 99, v: 0.06 },
    rho0: 1.0,
    u0: [0, 0],
  });
}

const SCENES = {
  dem_dam: { kind: "dem", build: demDamBreak },
  dem_column: { kind: "dem", build: demColumn },
  sph_dam: { kind: "sph", build: sphDamBreak },
  dem_cfd: { kind: "coupled", build: demCfdCoupledDem, buildCfd: demCfdCoupledCfd },
  cfd_cavity: { kind: "cfd", build: cfdCavity },
  cfd_cylinder: { kind: "cfd", build: cfdCylinder },
};

// ---------------------------------------------------------------------------
// WebGL helpers
// ---------------------------------------------------------------------------

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh));
  }
  return sh;
}

function program(gl, vsrc, fsrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p));
  }
  return p;
}

// ---------------------------------------------------------------------------
// DEM renderer — point-sprite spheres, colour by speed, slow auto-rotation
// ---------------------------------------------------------------------------

const DEM_VS = `
attribute vec2 a_pos;
attribute vec3 a_color;
attribute float a_size;
varying vec3 v_color;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
  gl_PointSize = a_size;
  v_color = a_color;
}`;

const DEM_FS = `
precision mediump float;
varying vec3 v_color;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(v_color, 1.0);
}`;

function makeDemRenderer(gl, canvas) {
  const prog = program(gl, DEM_VS, DEM_FS);
  const posBuf = gl.createBuffer();
  const colBuf = gl.createBuffer();
  const sizeBuf = gl.createBuffer();
  const a_pos = gl.getAttribLocation(prog, "a_pos");
  const a_color = gl.getAttribLocation(prog, "a_color");
  const a_size = gl.getAttribLocation(prog, "a_size");

  let angle = 0.6;

  function speedColor(v, maxV) {
    const t = Math.min(1, v / (maxV || 1));
    // blue -> cyan -> green -> yellow -> red ramp
    const r = Math.min(1, Math.max(0, 1.5 * t - 0.3));
    const g = Math.min(1, Math.max(0, 1.3 * t + 0.1));
    const b = Math.min(1, Math.max(0, 1.0 - 1.4 * t));
    return [r, g, b];
  }

  function frame(sim) {
    sim.step();
    const pos = sim.positions();
    const vel = sim.velocities();
    const n = sim.count();
    const w = canvas.width,
      h = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    const scale = (Math.min(w, h) / (5.0 * 2.2)) * dpr * 0.5;
    const cx = w / 2,
      cy = h / 2;
    const cos = Math.cos(angle),
      sin = Math.sin(angle);

    const screen = new Float32Array(n * 2);
    const colors = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    let maxV = 1e-6;
    for (let i = 0; i < n; i++) {
      const vx = vel[i * 3],
        vy = vel[i * 3 + 1],
        vz = vel[i * 3 + 2];
      const v = Math.hypot(vx, vy, vz);
      if (v > maxV) maxV = v;
    }
    for (let i = 0; i < n; i++) {
      const x = pos[i * 4],
        y = pos[i * 4 + 1],
        z = pos[i * 4 + 2],
        rad = pos[i * 4 + 3];
      const rx = x * cos + z * sin;
      const rz = -x * sin + z * cos;
      const sx = cx + rx * scale;
      const sy = cy - y * scale * 0.75 - rz * scale * 0.35;
      screen[i * 2] = (sx / w) * 2 - 1;
      screen[i * 2 + 1] = -((sy / h) * 2 - 1);
      const vx = vel[i * 3],
        vy = vel[i * 3 + 1],
        vz = vel[i * 3 + 2];
      const c = speedColor(Math.hypot(vx, vy, vz), maxV);
      colors[i * 3] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
      sizes[i] = Math.max(2, rad * scale * 2);
    }

    gl.viewport(0, 0, w, h);
    gl.clearColor(0.05, 0.06, 0.09, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, screen, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(a_pos);
    gl.vertexAttribPointer(a_pos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(a_color);
    gl.vertexAttribPointer(a_color, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, sizes, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(a_size);
    gl.vertexAttribPointer(a_size, 1, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.disable(gl.BLEND);

    angle += 0.004;
    return { kind: "DEM", n, ke: sim.kinetic_energy(), maxV };
  }

  return { frame };
}

// ---------------------------------------------------------------------------
// CFD renderer — speed field texture + velocity arrows
// ---------------------------------------------------------------------------

const FIELD_VS = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, a_pos.y * 0.5 + 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FIELD_FS = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_field;
vec3 ramp(float t) {
  t = clamp(t, 0.0, 1.0);
  return vec3(
    clamp(1.5 * t - 0.3, 0.0, 1.0),
    clamp(1.3 * t + 0.1, 0.0, 1.0),
    clamp(1.0 - 1.4 * t, 0.0, 1.0));
}
void main() {
  vec4 s = texture2D(u_field, v_uv);
  if (s.a < 0.5) { gl_FragColor = vec4(0.02, 0.02, 0.03, 1.0); return; }
  gl_FragColor = vec4(ramp(s.r), 1.0);
}`;

const ARROW_VS = `
attribute vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;
const ARROW_FS = `
precision mediump float;
void main() { gl_FragColor = vec4(0.9, 0.9, 0.95, 0.5); }`;

function makeCfdRenderer(gl, canvas) {
  const fieldProg = program(gl, FIELD_VS, FIELD_FS);
  const arrowProg = program(gl, ARROW_VS, ARROW_FS);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const tex = gl.createTexture();
  const arrowBuf = gl.createBuffer();
  const a_field = gl.getUniformLocation(fieldProg, "u_field");
  const a_pos_arrow = gl.getAttribLocation(arrowProg, "a_pos");

  function frame(sim) {
    sim.step();
    const nx = sim.nx(),
      ny = sim.ny();
    const vel = sim.velocity(); // [ux, uy, speed] * n
    const solid = sim.solid(); // u8 * n
    const w = canvas.width,
      h = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    // Pack speed into an RGBA texture; solid cells get alpha = 0 (black).
    const rgba = new Uint8Array(nx * ny * 4);
    let maxSpeed = 1e-6;
    for (let i = 0; i < nx * ny; i++) maxSpeed = Math.max(maxSpeed, vel[i * 3 + 2]);
    for (let i = 0; i < nx * ny; i++) {
      if (solid[i]) {
        rgba[i * 4 + 3] = 0;
        continue;
      }
      const t = vel[i * 3 + 2] / maxSpeed;
      rgba[i * 4] = Math.round(255 * Math.min(1, Math.max(0, 1.5 * t - 0.3)));
      rgba[i * 4 + 1] = Math.round(255 * Math.min(1, Math.max(0, 1.3 * t + 0.1)));
      rgba[i * 4 + 2] = Math.round(255 * Math.min(1, Math.max(0, 1.0 - 1.4 * t)));
      rgba[i * 4 + 3] = 255;
    }
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, nx, ny, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.viewport(0, 0, w, h);
    gl.clearColor(0.02, 0.02, 0.03, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Field quad.
    gl.useProgram(fieldProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(a_field, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    const fa = gl.getAttribLocation(fieldProg, "a_pos");
    gl.enableVertexAttribArray(fa);
    gl.vertexAttribPointer(fa, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Velocity arrows on a coarse grid.
    const stepX = Math.max(1, Math.floor(nx / 40));
    const stepY = Math.max(1, Math.floor(ny / 20));
    const arrowScale = (Math.min(w, h) / nx) * 0.9 * dpr * 0.9;
    const verts = [];
    for (let iy = stepY; iy < ny - stepY; iy += stepY) {
      for (let ix = stepX; ix < nx - stepX; ix += stepX) {
        const i = iy * nx + ix;
        if (solid[i]) continue;
        const ux = vel[i * 3],
          uy = vel[i * 3 + 1];
        if (ux * ux + uy * uy < 1e-12) continue;
        const x0 = (ix / (nx - 1)) * 2 - 1;
        const y0 = 1 - (iy / (ny - 1)) * 2;
        const x1 = x0 + ux * arrowScale;
        const y1 = y0 - uy * arrowScale;
        verts.push(x0, y0, x1, y1);
      }
    }
    gl.useProgram(arrowProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, arrowBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(a_pos_arrow);
    gl.vertexAttribPointer(a_pos_arrow, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, verts.length / 2);

    return { kind: "CFD", nx, ny, maxSpeed };
  }

  return { frame };
}

// ---------------------------------------------------------------------------
// SPH renderer — 2-D point sprites mapped from the domain box, coloured by
// speed and sized by the smoothing length `h`.
// ---------------------------------------------------------------------------

function makeSphRenderer(gl, canvas) {
  const prog = program(gl, DEM_VS, DEM_FS);
  const posBuf = gl.createBuffer();
  const colBuf = gl.createBuffer();
  const sizeBuf = gl.createBuffer();
  const a_pos = gl.getAttribLocation(prog, "a_pos");
  const a_color = gl.getAttribLocation(prog, "a_color");
  const a_size = gl.getAttribLocation(prog, "a_size");

  function speedColor(v, maxV) {
    const t = Math.min(1, v / (maxV || 1));
    const r = Math.min(1, Math.max(0, 1.5 * t - 0.3));
    const g = Math.min(1, Math.max(0, 1.3 * t + 0.1));
    const b = Math.min(1, Math.max(0, 1.0 - 1.4 * t));
    return [r, g, b];
  }

  function frame(sim) {
    sim.step();
    const n = sim.count();
    const pos = sim.positions(); // [x, y] * n
    const speeds = sim.speeds(); // speed * n
    const h = sim.smoothing_length();
    const w = canvas.width,
      h_px = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    // Domain is [0,1]x[0,1]; map to a centred square with small margin.
    const scale = (Math.min(w, h_px) * 0.9) / 2;
    const cx = w / 2,
      cy = h_px / 2;

    const screen = new Float32Array(n * 2);
    const colors = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    let maxV = 1e-6;
    for (let i = 0; i < n; i++) maxV = Math.max(maxV, speeds[i]);
    for (let i = 0; i < n; i++) {
      const x = pos[i * 2],
        y = pos[i * 2 + 1];
      screen[i * 2] = (x - 0.5) * 2 * (scale / (w / 2));
      screen[i * 2 + 1] = -(y - 0.5) * 2 * (scale / (h_px / 2));
      const c = speedColor(speeds[i], maxV);
      colors[i * 3] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
      sizes[i] = Math.max(2, (h / 0.5) * scale * dpr);
    }

    gl.viewport(0, 0, w, h_px);
    gl.clearColor(0.04, 0.05, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, screen, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(a_pos);
    gl.vertexAttribPointer(a_pos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(a_color);
    gl.vertexAttribPointer(a_color, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, sizes, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(a_size);
    gl.vertexAttribPointer(a_size, 1, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.disable(gl.BLEND);

    return { kind: "SPH", n, ke: sim.kinetic_energy(), maxV };
  }

  return { frame };
}

// ---------------------------------------------------------------------------
// Coupled DEM+CFD renderer — runs the DEM bed and the LBM cavity in one loop,
// drawing the granular phase and reporting both engines' health. One-way demo;
// the resolved two-way coupling lives in the Rust example `coupled_dem_cfd.rs`.
// ---------------------------------------------------------------------------

function makeCoupledRenderer(gl, canvas, cfdSim) {
  const demR = makeDemRenderer(gl, canvas);

  function frame(demSim) {
    cfdSim.step();
    const r = demR.frame(demSim);
    const vel = cfdSim.velocity(); // [ux, uy, speed] * n
    let cfdVmax = 0;
    for (let i = 0; i < vel.length; i += 3) cfdVmax = Math.max(cfdVmax, vel[i * 3 + 2]);
    return { kind: "DEM+CFD", n: r.n, ke: r.ke, maxV: r.maxV, cfdVmax };
  }

  return { frame };
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

async function main() {
  await init();
  const canvas = document.getElementById("view");
  const gl =
    canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
  if (!gl) {
    document.getElementById("stats").textContent = "WebGL not available";
    return;
  }

  const sceneSel = document.getElementById("scene");
  const playBtn = document.getElementById("play");
  const stepBtn = document.getElementById("step");
  const resetBtn = document.getElementById("reset");
  const statsEl = document.getElementById("stats");

  let sim = null;
  let renderer = null;
  let running = true;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
  }

  function loadScene() {
    resize();
    const spec = SCENES[sceneSel.value];
    const json = spec.build();
    if (spec.kind === "dem") {
      sim = new DemSimulation(json);
      renderer = makeDemRenderer(gl, canvas);
    } else if (spec.kind === "sph") {
      sim = new SphSimulation(json);
      renderer = makeSphRenderer(gl, canvas);
    } else if (spec.kind === "coupled") {
      sim = new DemSimulation(spec.build());
      const cfdSim = new CfdSimulation(spec.buildCfd());
      renderer = makeCoupledRenderer(gl, canvas, cfdSim);
    } else {
      sim = new CfdSimulation(json);
      renderer = makeCfdRenderer(gl, canvas);
    }
  }

  sceneSel.addEventListener("change", loadScene);
  resetBtn.addEventListener("click", loadScene);
  playBtn.addEventListener("click", () => {
    running = !running;
    playBtn.textContent = running ? "Pause" : "Play";
  });
  stepBtn.addEventListener("click", () => {
    if (sim && renderer) statsEl.textContent = fmt(renderer.frame(sim));
  });

  function fmt(r) {
    if (r.kind === "DEM")
      return `DEM  particles=${r.n}  KE=${r.ke.toFixed(2)} J  vmax=${r.maxV.toFixed(3)} m/s`;
    if (r.kind === "SPH")
      return `SPH  particles=${r.n}  KE=${r.ke.toFixed(2)} J  vmax=${r.maxV.toFixed(3)} m/s`;
    if (r.kind === "DEM+CFD")
      return `DEM+CFD  particles=${r.n}  KE=${r.ke.toFixed(2)} J  vmax=${r.maxV.toFixed(3)} m/s  cfdVmax=${r.cfdVmax.toFixed(4)} (lu)`;
    return `CFD  grid=${r.nx}x${r.ny}  vmax=${r.maxSpeed.toFixed(4)} (lu)`;
  }

  loadScene();

  function loop() {
    if (running && sim && renderer) {
      statsEl.textContent = fmt(renderer.frame(sim));
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((e) => {
  document.getElementById("stats").textContent = "error: " + e;
  console.error(e);
});
