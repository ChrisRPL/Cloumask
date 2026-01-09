# Point Cloud Viewer

> **Status:** 🔴 Not Started
> **Priority:** P2 (Medium - 3D data visualization)
> **Dependencies:** 01-design-system, 02-core-layout, 05-point-cloud (backend integration)
> **Estimated Complexity:** Very High

## Overview

Implement a Three.js-based 3D point cloud viewer for LiDAR data visualization. Supports loading, rendering, and navigating large point clouds with detection overlays and 2D-3D fusion capabilities.

## Goals

- [ ] Load and render point cloud files (LAS, LAZ, PCD, PLY)
- [ ] Orbit/pan/zoom camera controls
- [ ] Color by intensity, height, classification, or RGB
- [ ] 3D bounding box visualization
- [ ] Point picking and selection
- [ ] 2D-3D fusion (project camera images onto point cloud)
- [ ] Level-of-detail rendering for large datasets
- [ ] Measurement tools (distance, area)

## Technical Design

### Point Cloud Viewer Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  VIEWER HEADER                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ☁ Point Cloud Viewer              [Load] [Export] [Settings]     │  │
│  │  scene_001.las • 12.4M points • 1.2 GB                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  TOOLBAR                                                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  [◎ Orbit] [✥ Pan] [🔍 Zoom] [📏 Measure] │ Color: [Height ▼]   │  │
│  │  [📷 Screenshot] [⊞ Split View] [🎯 Focus] │ Size: [●────] 2px   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  3D VIEWPORT                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │                    . . . . . .                                    │  │
│  │                 . . . . . . . . .                                 │  │
│  │              ┌───────────────────┐                                │  │
│  │              │                   │  ← 3D Bounding Box             │  │
│  │           . .│   . . . . . .    │. .                              │  │
│  │          . . │  . . . . . . .   │ . .                             │  │
│  │         . . .│ . . . . . . . .  │. . .                            │  │
│  │              │                   │                                │  │
│  │              └───────────────────┘                                │  │
│  │           . . . . . . . . . . . . .                               │  │
│  │         . . . . . . . . . . . . . . .                             │  │
│  │                                                                   │  │
│  │  ──────────────────────                                           │  │
│  │         Ground Plane                                              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  INFO PANEL (collapsible)                                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Camera: (12.4, 5.2, 8.1) • Target: (0, 0, 0) • FPS: 60           │  │
│  │  Selection: 1,234 points • Class: Vehicle • Confidence: 0.89     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Three.js Architecture

```typescript
// Core rendering components
interface PointCloudScene {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  pointCloud: THREE.Points | null;
  boundingBoxes: THREE.LineSegments[];
  raycaster: THREE.Raycaster;
}

// Point cloud data structure
interface PointCloudData {
  positions: Float32Array;      // xyz coordinates
  colors: Float32Array;         // rgb values (0-1)
  intensities: Float32Array;    // intensity values
  classifications: Uint8Array;  // class labels
  count: number;
  bounds: {
    min: THREE.Vector3;
    max: THREE.Vector3;
  };
}

// Color mapping modes
type ColorMode =
  | 'rgb'            // Original RGB
  | 'intensity'      // Grayscale intensity
  | 'height'         // Rainbow by Z value
  | 'classification' // Categorical colors
  | 'custom';        // User-defined

// 3D Bounding box
interface BoundingBox3D {
  id: string;
  class_name: string;
  confidence: number;
  center: THREE.Vector3;
  size: THREE.Vector3;
  rotation: THREE.Euler;
  color: string;
}
```

### Component Hierarchy

```
PointCloudViewer.svelte
├── ViewerHeader.svelte
│   ├── FileInfo
│   └── ActionButtons (Load, Export, Settings)
├── ViewerToolbar.svelte
│   ├── NavigationTools (Orbit, Pan, Zoom)
│   ├── MeasurementTools
│   ├── ColorModeSelector
│   └── PointSizeSlider
├── ThreeCanvas.svelte
│   └── [Three.js WebGL Canvas]
├── InfoPanel.svelte
│   ├── CameraInfo
│   └── SelectionInfo
└── ViewerSettings.svelte (modal)
    ├── RenderingSettings
    ├── ColorSettings
    └── PerformanceSettings
```

## Implementation Tasks

- [ ] **Three.js Setup**
  - [ ] Create `src/lib/utils/three/setup.ts`
  - [ ] Initialize WebGL renderer
  - [ ] Configure camera (perspective, FOV 60)
  - [ ] Set up OrbitControls
  - [ ] Add lighting (ambient + directional)
  - [ ] Implement render loop with requestAnimationFrame

- [ ] **Point Cloud Loading**
  - [ ] Create `src/lib/utils/three/loaders.ts`
  - [ ] Integrate potree or pnext/three-loader
  - [ ] Load LAS/LAZ files via Tauri IPC
  - [ ] Load PCD files
  - [ ] Load PLY files
  - [ ] Progress callback for large files

- [ ] **Point Rendering**
  - [ ] Create custom shader material for points
  - [ ] Implement point size attenuation
  - [ ] Add color mode switching
  - [ ] Implement height-based coloring
  - [ ] Implement classification coloring
  - [ ] Handle intensity normalization

- [ ] **Camera Controls**
  - [ ] Orbit mode (default)
  - [ ] Pan mode
  - [ ] First-person fly mode
  - [ ] Focus on selection
  - [ ] Reset camera view
  - [ ] Keyboard shortcuts for navigation

- [ ] **ThreeCanvas Component**
  - [ ] Create `ThreeCanvas.svelte`
  - [ ] Manage canvas element lifecycle
  - [ ] Handle resize events
  - [ ] Mouse event handling
  - [ ] Touch support for tablets
  - [ ] Context menu prevention

- [ ] **3D Bounding Boxes**
  - [ ] Create `src/lib/utils/three/boxes.ts`
  - [ ] Render wireframe boxes
  - [ ] Add labels (class + confidence)
  - [ ] Color-code by class
  - [ ] Selection highlight
  - [ ] Show/hide toggle

- [ ] **Point Picking**
  - [ ] Implement raycasting for point selection
  - [ ] Highlight selected points
  - [ ] Display point info (xyz, class, intensity)
  - [ ] Box selection mode
  - [ ] Multi-select with Shift

- [ ] **Measurement Tools**
  - [ ] Distance measurement (point to point)
  - [ ] Area measurement (polygon)
  - [ ] Angle measurement
  - [ ] Display measurements in 3D
  - [ ] Unit conversion (m, ft)

- [ ] **Level of Detail (LOD)**
  - [ ] Implement octree-based culling
  - [ ] Dynamic point budget
  - [ ] Progressive loading
  - [ ] Frustum culling
  - [ ] Distance-based detail

- [ ] **ViewerToolbar Component**
  - [ ] Create `ViewerToolbar.svelte`
  - [ ] Tool selection buttons
  - [ ] Color mode dropdown
  - [ ] Point size slider
  - [ ] Screenshot button
  - [ ] Split view toggle

- [ ] **ViewerSettings Modal**
  - [ ] Create `ViewerSettings.svelte`
  - [ ] Background color picker
  - [ ] Axis helper toggle
  - [ ] Grid toggle
  - [ ] Point budget slider
  - [ ] EDL (Eye-Dome Lighting) toggle

- [ ] **2D-3D Fusion (Future)**
  - [ ] Camera calibration input
  - [ ] Image projection onto points
  - [ ] Split-screen 2D/3D view
  - [ ] Synchronized navigation

## Acceptance Criteria

- [ ] Point cloud loads and renders smoothly
- [ ] Camera controls work (orbit, pan, zoom)
- [ ] Color modes switch correctly
- [ ] 3D bounding boxes display with labels
- [ ] Point picking returns correct data
- [ ] Measurements work accurately
- [ ] FPS stays above 30 for 10M+ points
- [ ] Memory usage stays reasonable (<2GB)

## Files to Create/Modify

```
src/lib/components/PointCloud/
├── PointCloudViewer.svelte
├── ViewerHeader.svelte
├── ViewerToolbar.svelte
├── ThreeCanvas.svelte
├── InfoPanel.svelte
├── ViewerSettings.svelte
└── index.ts

src/lib/utils/three/
├── setup.ts           # Scene, camera, renderer setup
├── loaders.ts         # Point cloud file loaders
├── materials.ts       # Custom shader materials
├── boxes.ts           # 3D bounding box utilities
├── picking.ts         # Raycasting and selection
├── measurement.ts     # Measurement tools
├── lod.ts             # Level of detail management
└── index.ts
```

## Three.js Initialization

```typescript
// src/lib/utils/three/setup.ts
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export interface SceneContext {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  dispose: () => void;
}

export function createScene(canvas: HTMLCanvasElement): SceneContext {
  // Scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0b);

  // Camera
  const camera = new THREE.PerspectiveCamera(
    60,
    canvas.clientWidth / canvas.clientHeight,
    0.1,
    10000
  );
  camera.position.set(50, 50, 50);

  // Renderer
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);

  // Controls
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = true;

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const directional = new THREE.DirectionalLight(0xffffff, 0.5);
  directional.position.set(100, 100, 100);
  scene.add(directional);

  // Grid helper
  const grid = new THREE.GridHelper(100, 100, 0x404040, 0x303030);
  scene.add(grid);

  // Axis helper
  const axes = new THREE.AxesHelper(10);
  scene.add(axes);

  // Render loop
  let animationId: number;
  function animate() {
    animationId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Resize handler
  const resizeObserver = new ResizeObserver(() => {
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  });
  resizeObserver.observe(canvas);

  return {
    scene,
    camera,
    renderer,
    controls,
    dispose: () => {
      cancelAnimationFrame(animationId);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
    },
  };
}
```

## Point Cloud Shader

```glsl
// Vertex shader
uniform float pointSize;
uniform float heightMin;
uniform float heightMax;
attribute float intensity;

varying vec3 vColor;

vec3 heightColor(float height) {
  float t = clamp((height - heightMin) / (heightMax - heightMin), 0.0, 1.0);
  // Rainbow gradient
  vec3 a = vec3(0.0, 0.0, 1.0); // Blue
  vec3 b = vec3(0.0, 1.0, 1.0); // Cyan
  vec3 c = vec3(0.0, 1.0, 0.0); // Green
  vec3 d = vec3(1.0, 1.0, 0.0); // Yellow
  vec3 e = vec3(1.0, 0.0, 0.0); // Red

  if (t < 0.25) return mix(a, b, t * 4.0);
  if (t < 0.5) return mix(b, c, (t - 0.25) * 4.0);
  if (t < 0.75) return mix(c, d, (t - 0.5) * 4.0);
  return mix(d, e, (t - 0.75) * 4.0);
}

void main() {
  vColor = heightColor(position.z);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = pointSize * (300.0 / -mvPosition.z);
  gl_Position = projectionMatrix * mvPosition;
}

// Fragment shader
varying vec3 vColor;

void main() {
  // Circular point shape
  vec2 center = gl_PointCoord - vec2(0.5);
  if (length(center) > 0.5) discard;

  gl_FragColor = vec4(vColor, 1.0);
}
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Orbit mode |
| `2` | Pan mode |
| `3` | First-person mode |
| `R` | Reset camera |
| `F` | Focus on selection |
| `M` | Measure tool |
| `G` | Toggle grid |
| `A` | Toggle axes |
| `B` | Toggle bounding boxes |
| `+` / `=` | Increase point size |
| `-` | Decrease point size |
| `C` | Cycle color mode |

## Performance Considerations

- Use `BufferGeometry` with typed arrays
- Implement frustum culling
- Use octree for spatial queries
- Limit rendered points via LOD
- Use Web Workers for data parsing
- Consider GPU-based selection

## Notes

- Three.js r170+ required for modern features
- Test with KITTI, nuScenes datasets
- Consider WebGPU renderer for future
- Mobile: reduce point budget significantly
- Add export to PLY/PCD format
