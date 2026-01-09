# Three.js Point Cloud Viewer

> **Status:** 🔴 Not Started
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation (Svelte 5), 01-rust-io
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Frontend visualization component using Three.js for rendering point clouds with multiple color modes, 3D bounding box overlays, interactive camera controls, and Level of Detail (LOD) for handling large point clouds (10M+ points) at 60fps.

## Goals

- [ ] Render point clouds using Three.js Points geometry
- [ ] Support color modes: intensity gradient, height gradient, RGB passthrough
- [ ] Visualize 3D bounding boxes with class labels
- [ ] Implement orbit/pan/zoom camera controls
- [ ] LOD system using octree for 10M+ point clouds
- [ ] Svelte 5 component with reactive props

## Technical Design

### Dependencies

```json
// package.json
{
  "dependencies": {
    "three": "^0.170.0",
    "@types/three": "^0.170.0"
  }
}
```

### Component Architecture

```
src/lib/components/PointCloud/
├── index.ts           # Module exports
├── Viewer.svelte      # Main viewer component
├── Controls.svelte    # UI controls panel
├── renderer.ts        # Three.js scene setup
├── materials.ts       # Point cloud shaders
├── lod.ts             # LOD/octree system
├── bbox.ts            # 3D bounding box rendering
└── types.ts           # TypeScript interfaces
```

### TypeScript Interfaces

```typescript
// types.ts

export interface PointCloudData {
  positions: Float32Array;  // Flat [x, y, z, x, y, z, ...]
  colors?: Uint8Array;      // Flat [r, g, b, r, g, b, ...] 0-255
  intensity?: Float32Array; // [i0, i1, i2, ...]
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  pointCount: number;
}

export interface BBox3D {
  center: [number, number, number];
  dimensions: [number, number, number];  // length, width, height
  rotation: number;  // yaw in radians
  className: string;
  score: number;
  color?: string;  // CSS color for box wireframe
}

export type ColorMode = 'intensity' | 'height' | 'rgb' | 'uniform';

export interface ViewerSettings {
  colorMode: ColorMode;
  pointSize: number;
  backgroundColor: string;
  showBoundingBoxes: boolean;
  showLabels: boolean;
  lodEnabled: boolean;
}
```

### Three.js Renderer

```typescript
// renderer.ts
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class PointCloudRenderer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private pointCloud: THREE.Points | null = null;
  private bboxGroup: THREE.Group;

  constructor(container: HTMLElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a1a);

    // Camera setup
    this.camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 0, 50);

    // Renderer setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // Bounding box group
    this.bboxGroup = new THREE.Group();
    this.scene.add(this.bboxGroup);

    // Grid helper
    const gridHelper = new THREE.GridHelper(100, 100);
    this.scene.add(gridHelper);

    // Start render loop
    this.animate();
  }

  setPointCloud(data: PointCloudData, colorMode: ColorMode = 'height'): void {
    // Remove existing point cloud
    if (this.pointCloud) {
      this.scene.remove(this.pointCloud);
      this.pointCloud.geometry.dispose();
    }

    // Create geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(data.positions, 3)
    );

    // Set colors based on mode
    const colors = this.computeColors(data, colorMode);
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    // Create material
    const material = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      sizeAttenuation: true,
    });

    // Create points
    this.pointCloud = new THREE.Points(geometry, material);
    this.scene.add(this.pointCloud);

    // Center camera on point cloud
    this.centerOnBounds(data.bounds);
  }

  private computeColors(
    data: PointCloudData,
    mode: ColorMode
  ): Float32Array {
    const colors = new Float32Array(data.pointCount * 3);

    switch (mode) {
      case 'height':
        // Color by Z coordinate (blue to red)
        const zMin = data.bounds.min[2];
        const zRange = data.bounds.max[2] - zMin;
        for (let i = 0; i < data.pointCount; i++) {
          const z = data.positions[i * 3 + 2];
          const t = (z - zMin) / zRange;
          // Blue to red gradient
          colors[i * 3] = t;         // R
          colors[i * 3 + 1] = 0.2;   // G
          colors[i * 3 + 2] = 1 - t; // B
        }
        break;

      case 'intensity':
        if (!data.intensity) {
          return this.computeColors(data, 'uniform');
        }
        // Grayscale from intensity
        const iMin = Math.min(...data.intensity);
        const iMax = Math.max(...data.intensity);
        const iRange = iMax - iMin || 1;
        for (let i = 0; i < data.pointCount; i++) {
          const t = (data.intensity[i] - iMin) / iRange;
          colors[i * 3] = t;
          colors[i * 3 + 1] = t;
          colors[i * 3 + 2] = t;
        }
        break;

      case 'rgb':
        if (!data.colors) {
          return this.computeColors(data, 'uniform');
        }
        // Use provided RGB colors (normalize from 0-255 to 0-1)
        for (let i = 0; i < data.pointCount; i++) {
          colors[i * 3] = data.colors[i * 3] / 255;
          colors[i * 3 + 1] = data.colors[i * 3 + 1] / 255;
          colors[i * 3 + 2] = data.colors[i * 3 + 2] / 255;
        }
        break;

      case 'uniform':
      default:
        // Single color (white)
        colors.fill(1.0);
        break;
    }

    return colors;
  }

  setBoundingBoxes(boxes: BBox3D[]): void {
    // Clear existing boxes
    this.bboxGroup.clear();

    for (const box of boxes) {
      const mesh = this.createBBoxMesh(box);
      this.bboxGroup.add(mesh);
    }
  }

  private createBBoxMesh(box: BBox3D): THREE.Object3D {
    const group = new THREE.Group();

    // Wireframe box
    const geometry = new THREE.BoxGeometry(...box.dimensions);
    const edges = new THREE.EdgesGeometry(geometry);
    const material = new THREE.LineBasicMaterial({
      color: box.color || this.getClassColor(box.className),
    });
    const wireframe = new THREE.LineSegments(edges, material);

    // Position and rotate
    wireframe.position.set(...box.center);
    wireframe.rotation.z = box.rotation;

    group.add(wireframe);

    // TODO: Add label sprite with className and score

    return group;
  }

  private getClassColor(className: string): number {
    const colors: Record<string, number> = {
      'Car': 0x00ff00,
      'Pedestrian': 0xff0000,
      'Cyclist': 0x0000ff,
      'Truck': 0xffff00,
    };
    return colors[className] || 0xffffff;
  }

  private centerOnBounds(bounds: PointCloudData['bounds']): void {
    const center = new THREE.Vector3(
      (bounds.min[0] + bounds.max[0]) / 2,
      (bounds.min[1] + bounds.max[1]) / 2,
      (bounds.min[2] + bounds.max[2]) / 2
    );
    this.controls.target.copy(center);

    const size = Math.max(
      bounds.max[0] - bounds.min[0],
      bounds.max[1] - bounds.min[1],
      bounds.max[2] - bounds.min[2]
    );
    this.camera.position.set(
      center.x + size,
      center.y + size,
      center.z + size
    );
    this.controls.update();
  }

  setPointSize(size: number): void {
    if (this.pointCloud) {
      (this.pointCloud.material as THREE.PointsMaterial).size = size;
    }
  }

  private animate = (): void => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  dispose(): void {
    this.renderer.dispose();
    this.controls.dispose();
    if (this.pointCloud) {
      this.pointCloud.geometry.dispose();
      (this.pointCloud.material as THREE.Material).dispose();
    }
  }
}
```

### Svelte 5 Component

```svelte
<!-- Viewer.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { PointCloudRenderer } from './renderer';
  import type { PointCloudData, BBox3D, ColorMode, ViewerSettings } from './types';

  // Props using Svelte 5 runes
  let {
    data = $bindable<PointCloudData | null>(null),
    boxes = $bindable<BBox3D[]>([]),
    settings = $bindable<ViewerSettings>({
      colorMode: 'height',
      pointSize: 0.05,
      backgroundColor: '#1a1a1a',
      showBoundingBoxes: true,
      showLabels: true,
      lodEnabled: true,
    }),
  } = $props();

  let container: HTMLDivElement;
  let renderer: PointCloudRenderer | null = null;

  onMount(() => {
    renderer = new PointCloudRenderer(container);
  });

  onDestroy(() => {
    renderer?.dispose();
  });

  // React to data changes
  $effect(() => {
    if (renderer && data) {
      renderer.setPointCloud(data, settings.colorMode);
    }
  });

  // React to box changes
  $effect(() => {
    if (renderer && settings.showBoundingBoxes) {
      renderer.setBoundingBoxes(boxes);
    }
  });

  // React to settings changes
  $effect(() => {
    if (renderer) {
      renderer.setPointSize(settings.pointSize);
    }
  });
</script>

<div class="viewer-container" bind:this={container}></div>

<style>
  .viewer-container {
    width: 100%;
    height: 100%;
    min-height: 400px;
  }
</style>
```

### LOD System

```typescript
// lod.ts
import * as THREE from 'three';

interface OctreeNode {
  bounds: THREE.Box3;
  points: Float32Array;
  children: OctreeNode[];
  level: number;
}

export class PointCloudLOD {
  private root: OctreeNode | null = null;
  private maxDepth: number = 8;
  private maxPointsPerNode: number = 10000;

  build(positions: Float32Array, bounds: THREE.Box3): void {
    this.root = this.buildNode(positions, bounds, 0);
  }

  private buildNode(
    positions: Float32Array,
    bounds: THREE.Box3,
    level: number
  ): OctreeNode {
    const pointCount = positions.length / 3;

    if (pointCount <= this.maxPointsPerNode || level >= this.maxDepth) {
      return {
        bounds,
        points: positions,
        children: [],
        level,
      };
    }

    // Subdivide into 8 children
    const children: OctreeNode[] = [];
    const childBounds = this.subdivide(bounds);

    for (const childBound of childBounds) {
      const childPoints = this.filterPoints(positions, childBound);
      if (childPoints.length > 0) {
        children.push(this.buildNode(childPoints, childBound, level + 1));
      }
    }

    // Store decimated points for this level
    const decimated = this.decimate(positions, this.maxPointsPerNode);

    return {
      bounds,
      points: decimated,
      children,
      level,
    };
  }

  getVisiblePoints(
    camera: THREE.Camera,
    targetPointCount: number
  ): Float32Array {
    if (!this.root) return new Float32Array(0);

    const frustum = new THREE.Frustum();
    frustum.setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(
        camera.projectionMatrix,
        camera.matrixWorldInverse
      )
    );

    const visiblePoints: Float32Array[] = [];
    this.collectVisible(this.root, frustum, camera, visiblePoints, targetPointCount);

    return this.concatenate(visiblePoints);
  }

  private collectVisible(
    node: OctreeNode,
    frustum: THREE.Frustum,
    camera: THREE.Camera,
    result: Float32Array[],
    budget: number
  ): void {
    if (!frustum.intersectsBox(node.bounds)) return;

    // Calculate screen-space size to determine LOD level
    const screenSize = this.computeScreenSize(node.bounds, camera);

    if (node.children.length === 0 || screenSize < 100) {
      // Use this node's points
      result.push(node.points);
    } else {
      // Recurse into children
      for (const child of node.children) {
        this.collectVisible(child, frustum, camera, result, budget);
      }
    }
  }

  // ... helper methods
}
```

## Implementation Tasks

- [ ] **Setup Three.js**
  - [ ] Add three.js to package.json
  - [ ] Configure TypeScript types
  - [ ] Create component directory structure

- [ ] **Implement PointCloudRenderer**
  - [ ] Scene, camera, renderer setup
  - [ ] OrbitControls integration
  - [ ] Point cloud geometry creation
  - [ ] Render loop with animation

- [ ] **Implement color modes**
  - [ ] Height-based gradient
  - [ ] Intensity-based grayscale
  - [ ] RGB passthrough
  - [ ] Uniform color fallback

- [ ] **Implement bounding boxes**
  - [ ] Wireframe box geometry
  - [ ] Class-based coloring
  - [ ] Label sprites with text

- [ ] **Implement LOD system**
  - [ ] Octree construction
  - [ ] Frustum culling
  - [ ] Screen-space LOD selection
  - [ ] Progressive streaming

- [ ] **Svelte 5 integration**
  - [ ] Create Viewer.svelte
  - [ ] Create Controls.svelte
  - [ ] Reactive updates with $effect
  - [ ] Cleanup on destroy

- [ ] **Performance optimization**
  - [ ] Use BufferGeometry
  - [ ] Instanced rendering for boxes
  - [ ] WebWorker for octree building

## Files to Create/Modify

| Path | Action | Purpose |
|------|--------|---------|
| `package.json` | Modify | Add three.js dependency |
| `src/lib/components/PointCloud/index.ts` | Create | Module exports |
| `src/lib/components/PointCloud/Viewer.svelte` | Create | Main viewer |
| `src/lib/components/PointCloud/Controls.svelte` | Create | UI controls |
| `src/lib/components/PointCloud/renderer.ts` | Create | Three.js setup |
| `src/lib/components/PointCloud/materials.ts` | Create | Custom shaders |
| `src/lib/components/PointCloud/lod.ts` | Create | LOD system |
| `src/lib/components/PointCloud/bbox.ts` | Create | Bounding box rendering |
| `src/lib/components/PointCloud/types.ts` | Create | TypeScript interfaces |

## API Reference

### Viewer Component Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `data` | `PointCloudData \| null` | `null` | Point cloud to display |
| `boxes` | `BBox3D[]` | `[]` | 3D bounding boxes to overlay |
| `settings` | `ViewerSettings` | `{...}` | Viewer configuration |

### Events

| Event | Payload | Description |
|-------|---------|-------------|
| `pointclick` | `{ index: number, position: [x, y, z] }` | Point clicked |
| `boxclick` | `{ box: BBox3D }` | Bounding box clicked |
| `camerachange` | `{ position, target }` | Camera moved |

## Acceptance Criteria

- [ ] Render 1M points at 60fps on integrated GPU
- [ ] Color mode switching between intensity/height/RGB in <16ms
- [ ] 3D bounding boxes display with class labels readable at all angles
- [ ] LOD system maintains 60fps for 10M+ point clouds
- [ ] Orbit controls feel smooth (damping, no lag)
- [ ] Component properly disposes Three.js resources on unmount
- [ ] `npm run check` passes for all TypeScript
- [ ] `npm run test` passes for component tests

## Testing Strategy

```typescript
import { render, fireEvent } from '@testing-library/svelte';
import Viewer from './Viewer.svelte';

describe('PointCloud Viewer', () => {
  it('renders empty state', async () => {
    const { container } = render(Viewer);
    expect(container.querySelector('.viewer-container')).toBeTruthy();
  });

  it('updates on data change', async () => {
    const { component } = render(Viewer);
    const data: PointCloudData = {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      pointCount: 2,
      bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    };
    await component.$set({ data });
    // Verify renderer was called
  });
});
```

## Performance Benchmarks

| Scenario | Target | Notes |
|----------|--------|-------|
| 100K points, integrated GPU | 60fps | Baseline |
| 1M points, integrated GPU | 60fps | With LOD |
| 10M points, dedicated GPU | 60fps | Full LOD |
| Color mode switch | <16ms | No geometry rebuild |
| Initial load, 1M points | <500ms | Including buffer upload |

## Related Sub-Specs

- [01-rust-io.md](./01-rust-io.md) - Streams point data to viewer
- [04-3d-detection.md](./04-3d-detection.md) - Source of 3D bounding boxes
- [08-agent-tools.md](./08-agent-tools.md) - Viewer triggered by agent
