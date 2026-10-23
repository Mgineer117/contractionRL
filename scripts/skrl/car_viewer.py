"""Display-only ground and episode paths for single-car Isaac playback."""

import numpy as np
from pxr import Gf, UsdGeom, UsdLux, UsdShade, Vt


class CarViewer:
    def __init__(self, env, *, name="CarViewer", trace_color=(1., .4, .06), frame_camera=True):
        self.env = env
        self.stage = env.sim.stage
        self.path = "/World/" + name
        self.frame_camera = frame_camera
        self.origin = env.scene.env_origins[0].cpu().numpy()
        self.previous_step = -1
        self.trace = []
        light = UsdLux.DomeLight.Get(self.stage, "/World/Light")
        light.GetColorAttr().Set(Gf.Vec3f(.22, .27, .35))
        light.GetIntensityAttr().Set(900.)
        self.ground = UsdGeom.Cube.Define(self.stage, self.path + "/Ground")
        self.ground.CreateSizeAttr(1.)
        self.ground_position = self.ground.AddTranslateOp()
        self.ground_size = self.ground.AddScaleOp()
        self._material(self.ground, "Ground", (.14, .17, .22), (.035, .045, .06))
        self.grid = self._curve("Grid", (.28, .34, .42), .025)
        self.reference = self._curve("ReferencePath", (.15, 1., .3), .12)
        self.traveled = self._curve("CarTrace", trace_color, .09)
        self.update()

    def _material(self, geometry, name, color, emission):
        path = self.path + "/Materials/" + name
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        from pxr import Sdf
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emission)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)

    def _curve(self, name, color, width):
        curve = UsdGeom.BasisCurves.Define(self.stage, self.path + "/" + name)
        curve.CreateTypeAttr("linear")
        curve.CreateWrapAttr("nonperiodic")
        curve.CreateWidthsAttr([width])
        curve.SetWidthsInterpolation("constant")
        self._material(curve, name, color, color)
        return curve

    @staticmethod
    def _points(curve, points, counts=None):
        points = np.asarray(points, dtype=np.float32)
        curve.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        curve.CreateCurveVertexCountsAttr().Set(counts if counts is not None else [len(points)])
        curve.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(Vt.Vec3fArray.FromNumpy(points)))

    def update(self):
        plant = self.env.plant
        step = int(plant.time_steps[0])
        current = plant.x_t[0, :2].cpu().numpy() + self.origin[:2]
        if step <= self.previous_step or self.previous_step < 0:
            offset = (plant.x_t[0, :2] - plant.xref[0, 0, :2]).tolist()
            print(f"[viewer] Episode start: position offset={offset} m; gap={np.linalg.norm(offset):.3f} m.", flush=True)
            gust = getattr(plant, "disturbance", None)
            if gust is not None:
                print(f"[viewer] Disturbance enabled={gust.enabled}; scales={gust.scales.tolist()}; sample={gust.value[0].tolist()}.", flush=True)
            self.trace = []
            length = round(plant.time_bound / plant.dt) + 1
            ref = plant.xref[0, :length, :2].cpu().numpy() + self.origin[:2]
            self._points(self.reference, np.column_stack((ref, np.full(len(ref), .045))))
            bounds = np.vstack((ref, current))
            low, high = bounds.min(0) - 8., bounds.max(0) + 8.
            center = (low + high) / 2
            size = np.maximum(high - low, 40.)
            self.ground_position.Set(Gf.Vec3d(float(center[0]), float(center[1]), -.06))
            self.ground_size.Set(Gf.Vec3d(float(size[0]), float(size[1]), .1))
            low, high = center - size / 2, center + size / 2
            lines = []
            for x in np.arange(np.ceil(low[0]), high[0], 1.):
                lines.extend(((x, low[1], .005), (x, high[1], .005)))
            for y in np.arange(np.ceil(low[1]), high[1], 1.):
                lines.extend(((low[0], y, .005), (high[0], y, .005)))
            self._points(self.grid, lines, [2] * (len(lines) // 2))
            span = max(float(np.ptp(bounds, axis=0).max()) + 8., 16.)
            if self.frame_camera:
                self.env.sim.set_camera_view(
                    (float(center[0]), float(center[1] - .55 * span), .95 * span),
                    (float(center[0]), float(center[1]), 0.),
                )
        self.trace.append((float(current[0]), float(current[1]), .07))
        self._points(self.traveled, self.trace if len(self.trace) > 1 else self.trace * 2)
        self.previous_step = step
