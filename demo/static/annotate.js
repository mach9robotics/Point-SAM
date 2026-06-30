import * as THREE from "three";
import { camera, points, renderer, origin_colors } from "/static/viewer.js";

var positive_flag = true;
var drag_flag = false;
var raycaster = new THREE.Raycaster();
var alpha = 0.4;
var mask_color = [0, 0, 1]; // segmented-mask highlight (blue)
var prompts = [];
var labels = [];

// ----- prompt mode (positive / negative) -------------------------------------
function setMode(positive) {
  positive_flag = positive;
  var pos = document.getElementById("annotate-positive");
  var neg = document.getElementById("annotate-negative");
  if (pos && neg) {
    // highlight the active mode so it's obvious which prompt you're placing
    pos.style.background = positive ? "#39d353" : "#ccc"; // green when active
    neg.style.background = positive ? "#ccc" : "#e5534b"; // red when active
  }
}
function onPositiveClick() {
  setMode(true);
}
function onNegativeClick() {
  setMode(false);
}

async function onSaveClick() {
  await fetch("/save", { method: "POST" });
  await reset();
}

async function onNextClick() {
  await fetch("/next", { method: "POST" });
  await reset();
}

async function reset() {
  var colors = points.geometry.attributes.color;
  for (var i = 0; i < origin_colors.count; i++) {
    colors.setXYZ(i, origin_colors.getX(i), origin_colors.getY(i), origin_colors.getZ(i));
  }
  colors.needsUpdate = true;
  prompts = [];
  labels = [];
  await fetch("/clear", { method: "POST" });
}

function bindButtons() {
  document.getElementById("annotate-negative").onclick = onNegativeClick;
  document.getElementById("annotate-positive").onclick = onPositiveClick;
  document.getElementById("save-result").onclick = onSaveClick;
  document.getElementById("clear-result").onclick = reset;
  document.getElementById("annotate-next").onclick = onNextClick;
}

function main() {
  bindButtons();
  setMode(true); // positive by default
  // Right-click is a shortcut for a negative prompt; stop the context menu so it
  // does not pop up over the canvas.
  renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());
}

// Pick the displayed point whose SCREEN projection is nearest the cursor.
//
// raycaster.intersectObject(points) returns every point within `threshold` world
// units of the ray, sorted by distance *from the camera* — so intersects[0] is the
// frontmost point in a cylinder around the ray, which is offset from where you
// clicked. Annotation tools instead pick the point that appears under the cursor.
// We re-rank the candidates by screen-space distance and take the nearest, growing
// the threshold only if nothing was hit (so empty space still snaps to a point).
function pickNearestPoint(mouse) {
  raycaster.setFromCamera(mouse, camera);
  var posAttr = points.geometry.attributes.position;
  var v = new THREE.Vector3();
  var thresholds = [0.05, 0.2, 0.6];
  for (var ti = 0; ti < thresholds.length; ti++) {
    raycaster.params.Points.threshold = thresholds[ti];
    var hits = raycaster.intersectObject(points);
    if (hits.length === 0) continue;
    var best = null;
    var bestD = Infinity;
    for (var i = 0; i < hits.length; i++) {
      v.fromBufferAttribute(posAttr, hits[i].index);
      v.applyMatrix4(points.matrixWorld).project(camera); // -> NDC
      var dx = v.x - mouse.x;
      var dy = v.y - mouse.y;
      var d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = hits[i];
      }
    }
    return best;
  }
  return null;
}

async function onMouseClick(event) {
  event.preventDefault();
  if (event.target !== renderer.domElement) {
    return;
  }

  // Left-click uses the current mode (set by the buttons); right-click is a quick
  // negative prompt. We do NOT force the mode from the mouse button anymore — that
  // was what made the "Negative Prompt" button do nothing.
  var label = event.button === 2 ? false : positive_flag;

  var rect = renderer.domElement.getBoundingClientRect();
  var x = (event.clientX - rect.left) / rect.width;
  var y = (event.clientY - rect.top) / rect.height;
  var mouse = new THREE.Vector2(x * 2 - 1, -(y * 2) + 1);

  var picked = pickNearestPoint(mouse);
  if (picked === null) {
    return; // click missed the point cloud
  }

  prompts.push(picked.index);
  labels.push(label);
  var response = await fetch("/segment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt_index: picked.index,
      prompt_label: label,
    }),
  });
  var data = await response.json();
  var mask = data.seg;

  // Alpha-blend the mask onto the point cloud
  var colors = points.geometry.attributes.color;
  for (var i = 0; i < mask.length; i++) {
    var ox = origin_colors.getX(i);
    var oy = origin_colors.getY(i);
    var oz = origin_colors.getZ(i);
    if (mask[i] > 0) {
      colors.setXYZ(
        i,
        ox * (1 - alpha) + mask_color[0] * alpha,
        oy * (1 - alpha) + mask_color[1] * alpha,
        oz * (1 - alpha) + mask_color[2] * alpha
      );
    } else {
      colors.setXYZ(i, ox, oy, oz);
    }
  }

  // Draw prompt markers (SAM convention: positive = green, negative = red)
  for (var i = 0; i < prompts.length; i++) {
    if (labels[i]) {
      colors.setXYZ(prompts[i], 0, 1, 0); // positive -> green
    } else {
      colors.setXYZ(prompts[i], 1, 0, 0); // negative -> red
    }
  }
  colors.needsUpdate = true;
}

window.addEventListener("mousedown", () => (drag_flag = false));
window.addEventListener("mousemove", () => (drag_flag = true));
window.addEventListener("mouseup", (event) => {
  if (!drag_flag) {
    onMouseClick(event);
  }
});

main();
