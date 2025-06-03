import viser
import math
from pathlib import Path
from typing import Literal
from typing import Tuple, Callable, List, Dict
from nerfview import Viewer, RenderTabState
import numpy as np
import viser.transforms as vtf

class GsplatRenderTabState(RenderTabState):
    # non-controllable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # existing controllable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = True
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # NEW — visibility toggle for camera frustums
    show_cameras: bool = False
    frustum_scale: float = 0.03

class GsplatViewer(Viewer):
    """
    Viewer for 3-D Gaussian Splatting **with camera-frustum visualisation**.

    Args
    ----
    server        : viser.ViserServer – the underlying server
    render_fn     : Callable           – your callable that does the splat rendering
    output_dir    : Path               – where screenshots / exports are saved
    mode          : "rendering" | "training"
    cameras       : optional list of dicts, one per image:
                    {
                        "fov":      float (vertical FoV in *radians*)
                        "aspect":   float (w / h)
                        "position": (x, y, z)
                        "wxyz":     quaternion as (w, x, y, z)
                    }
                    Feel free to add extra keys; only the four above are used.
    """

    # ------------------------------ INITIALISER ------------------------------ #
    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        cameras: List[Dict] | None = None,
    ):
        self._cameras: List[Dict] = cameras or []
        self._frustum_handles: List[viser.CameraFrustumHandle] = []

        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

        # draw the cameras *after* Viewer initialisation (so `server.scene` exists)
        if self._cameras:
            self._draw_camera_frustums()

    # -------------------------- GUI / TAB INITIALISATION -------------------- #
    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                max_sh_degree_number = server.gui.add_number(
                    "Max SH",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used",
                )

                @max_sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.max_sh_degree = int(
                        max_sh_degree_number.value
                    )
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to the egienvalues of projected 2D covariance matrices.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    if "depth" in render_mode_dropdown.value:
                        normalize_nearfar_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                    if render_mode_dropdown.value == "rgb":
                        inverse_checkbox.disabled = True
                    else:
                        inverse_checkbox.disabled = False
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
                    hint="Normalize depth with near/far plane.",
                )

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = (
                        normalize_nearfar_checkbox.value
                    )
                    self.rerender(_)

                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Whether to use classic or antialiased rasterization.",
                )

                @rasterize_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                    self.rerender(_)

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    self.rerender(_)

                # ───── NEW VISIBILITY TOGGLE FOR FRUSTUMS ───── #
                show_cameras_checkbox = server.gui.add_checkbox(
                    "Show Cameras",
                    initial_value=self.render_tab_state.show_cameras,
                    hint="Toggle camera-frustum visualisation.",
                )

                frustum_scale_slider = server.gui.add_slider(            # ← new control
                    "Frustum Scale",
                    min=0.001,
                    max=100,
                    step=0.001,
                    initial_value=self.render_tab_state.frustum_scale,
                    hint="Size of each camera frustum (world units).",
                )

                @show_cameras_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.show_cameras = show_cameras_checkbox.value
                    for h in self._frustum_handles:
                        h.visible = show_cameras_checkbox.value

                @frustum_scale_slider.on_update
                def _(_):
                    self.render_tab_state.frustum_scale = frustum_scale_slider.value
                    for h in self._frustum_handles:           # live-resize every frustum
                        h.scale = frustum_scale_slider.value

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
                "show_cameras_checkbox": show_cameras_checkbox,
                "frustum_scale_slider": frustum_scale_slider,
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count

    # --------------------------------------------------------------------- #
    #                            CAMERA FRUSTUMS                            #
    # --------------------------------------------------------------------- #
    def _draw_camera_frustums(self) -> None:
        scene  = self.server.scene
        scale  = self.render_tab_state.frustum_scale      # <── use the state value
        for i, cam in enumerate(self._cameras):
            name = f"/cameras/cam_{i:04d}"

            h = scene.add_camera_frustum(
                name=f"{name}/frustum",
                fov=cam["fov"],
                aspect=cam["aspect"],
                scale=scale,                           # ← here too
                line_width=1.8,
                position=cam["position"],
                wxyz=cam["wxyz"],
                color=(255, 180, 0),
                visible=self.render_tab_state.show_cameras,
            )
            self._frustum_handles.append(h)

# -----------------------------------------------------------------------------#
#                        HELPER – BUILD CAMERA DICTS                           #
# -----------------------------------------------------------------------------#
def build_camera_dict(
    pose_world_cam: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float | None = None,
) -> Dict:
    """
    Convenience helper that converts COLMAP-style intrinsics & a 4×4 pose
    into the dictionary format expected by `GsplatViewer`.

    Args
    ----
    pose_world_cam : 4×4 ndarray – homogeneous transform (world ← cam)
    width, height  : int         – image resolution
    fx, fy         : focal lengths (pixels).  `fy` defaults to `fx`.
    """
    fy = fx if fy is None else fy
    fov = 2 * math.atan2(height / 2.0, fy)
    aspect = width / height

    # rotation quaternion (w, x, y, z) in world frame
    R = pose_world_cam[:3, :3]
    t = pose_world_cam[:3, 3]
    wxyz = vtf.SO3.from_matrix(R).wxyz  # viser’s quaternion helper

    wxyz   = tuple(float(v) for v in wxyz)         # <-- cast
    position = tuple(float(v) for v in t)        # <-- cast

    return dict(fov=fov, aspect=aspect, position=position, wxyz=wxyz)