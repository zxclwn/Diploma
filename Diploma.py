import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import ttk
import threading
import warnings
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import griddata
import time
from localization import tr, set_lang

warnings.filterwarnings("ignore", category=RuntimeWarning)

class DamSimulation:
    def __init__(self, Lx, Ly, k_filt, porosity, D_diff, abort_check=None):
        self.Lx, self.Ly = Lx, Ly
        self.k, self.n, self.D = k_filt, porosity, D_diff
        self.dx, self.dy = 0.5, 0.5
        self.history = []
        self.fem_nodes = None
        self.fem_elements = None
        self.fem_H = None
        self.fem_history = []
        self.abort_check = abort_check

    def init_grids(self):
        self.Nx, self.Ny = int(self.Lx / self.dx) + 1, int(self.Ly / self.dy) + 1
        self.grid = np.ones((self.Ny, self.Nx))
        self.H = np.zeros((self.Ny, self.Nx))
        self.C = np.zeros((self.Ny, self.Nx))
        self.Vx = np.zeros((self.Ny, self.Nx))
        self.Vy = np.zeros((self.Ny, self.Nx))

    def check_abort(self):
        if self.abort_check and self.abort_check():
            raise InterruptedError(tr("Розрахунок зупинено користувачем"))

    def set_dam_geometry(self, x_start, x_end, depth):
        self.dam_start_m = x_start
        self.dam_end_m = x_end
        self.dam_depth_m = depth

        self.dam_start_idx = int(x_start / self.dx)
        self.dam_end_idx = int(x_end / self.dx)
        self.dam_depth_idx = int(depth / self.dy)
        self.grid[0:self.dam_depth_idx, self.dam_start_idx:self.dam_end_idx] = 0

    def add_pile(self, x_pos, pile_len, thickness_m=0.5):
        ix_start = min(int(x_pos / self.dx), self.Nx)
        ix_end = min(ix_start + max(1, int(thickness_m / self.dx)), self.Nx)
        iy_start = self.dam_depth_idx
        iy_end = min(iy_start + int(pile_len / self.dy), self.Ny)
        self.grid[iy_start:iy_end, ix_start:ix_end] = 0

    def is_solid(self, x, y):
        eps = 1e-5
        if self.dam_start_m + eps < x < self.dam_end_m - eps and y < self.dam_depth_m - eps:
            return True
        for _, px, plen in self.p_list_data:
            if px + eps < x < px + 0.5 - eps and self.dam_depth_m - eps < y < self.dam_depth_m + plen - eps:
                return True
        return False

    def build_fem_mesh(self, elem_size):
        h_max = elem_size
        h_min = max(0.15, h_max / 6.0)
        growth_rate = 0.35

        feature_pts = []
        step = h_min / 2.0

        for y in np.arange(0, self.dam_depth_m + step, step):
            feature_pts.append([self.dam_start_m, y])
            feature_pts.append([self.dam_end_m, y])
        for x in np.arange(self.dam_start_m, self.dam_end_m + step, step):
            feature_pts.append([x, self.dam_depth_m])

        for _, px, plen in self.p_list_data:
            thick = 0.5
            for y in np.arange(self.dam_depth_m, self.dam_depth_m + plen + step, step):
                feature_pts.append([px, y])
                feature_pts.append([px + thick, y])
            for x in np.arange(px, px + thick + step, step):
                feature_pts.append([x, self.dam_depth_m + plen])

        feature_pts = np.array(feature_pts)
        feature_tree = cKDTree(feature_pts) if len(feature_pts) > 0 else None

        def get_h(x, y):
            if feature_tree is None: return h_max
            d, _ = feature_tree.query([x, y])
            return np.clip(h_min + growth_rate * d, h_min, h_max)

        b_nodes = []

        def add_line_adaptive(x0, y0, x1, y1):
            L = np.hypot(x1 - x0, y1 - y0)
            if L == 0: return
            t = 0
            while t < L:
                cx = x0 + (x1 - x0) * (t / L)
                cy = y0 + (y1 - y0) * (t / L)
                b_nodes.append([cx, cy])
                t += get_h(cx, cy)
            b_nodes.append([x1, y1])

        add_line_adaptive(0, 0, self.Lx, 0)
        add_line_adaptive(self.Lx, 0, self.Lx, self.Ly)
        add_line_adaptive(self.Lx, self.Ly, 0, self.Ly)
        add_line_adaptive(0, self.Ly, 0, 0)

        add_line_adaptive(self.dam_start_m, 0, self.dam_start_m, self.dam_depth_m)
        add_line_adaptive(self.dam_start_m, self.dam_depth_m, self.dam_end_m, self.dam_depth_m)
        add_line_adaptive(self.dam_end_m, self.dam_depth_m, self.dam_end_m, 0)

        for _, px, plen in self.p_list_data:
            thick = 0.5
            add_line_adaptive(px, self.dam_depth_m, px, self.dam_depth_m + plen)
            add_line_adaptive(px + thick, self.dam_depth_m, px + thick, self.dam_depth_m + plen)
            add_line_adaptive(px, self.dam_depth_m + plen, px + thick, self.dam_depth_m + plen)

        b_nodes = np.array(b_nodes)
        b_nodes = np.unique(np.round(b_nodes, decimals=4), axis=0)

        valid_b = []
        for bx, by in b_nodes:
            if not self.is_solid(bx, by):
                valid_b.append([bx, by])
        b_nodes = np.array(valid_b)

        ix_vals = np.arange(0, self.Lx, h_min * 0.7)
        iy_vals = np.arange(0, self.Ly, h_min * 0.7 * (np.sqrt(3) / 2))

        candidates = []
        for j, y in enumerate(iy_vals):
            for i, x in enumerate(ix_vals):
                cx = x + (h_min * 0.35 if j % 2 == 1 else 0)
                cx += np.random.uniform(-0.2, 0.2) * h_min
                cy = y + np.random.uniform(-0.2, 0.2) * h_min
                if 0 < cx < self.Lx and 0 < cy < self.Ly and not self.is_solid(cx, cy):
                    candidates.append([cx, cy])

        candidates = np.array(candidates)

        if len(candidates) > 0:
            dists, _ = feature_tree.query(candidates)
            h_locals = np.clip(h_min + growth_rate * dists, h_min, h_max)

            sort_idx = np.argsort(h_locals)
            candidates = candidates[sort_idx]
            h_locals = h_locals[sort_idx]

            cell_size = h_min / np.sqrt(2)
            grid_w = int(np.ceil(self.Lx / cell_size))
            grid_h = int(np.ceil(self.Ly / cell_size))
            occ_grid = [[[] for _ in range(grid_h)] for _ in range(grid_w)]

            i_nodes = []
            for bx, by in b_nodes:
                gx, gy = int(bx / cell_size), int(by / cell_size)
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    occ_grid[gx][gy].append([bx, by])

            for (cx, cy), h_loc in zip(candidates, h_locals):
                gx, gy = int(cx / cell_size), int(cy / cell_size)
                search_rad = int(np.ceil(h_loc * 0.85 / cell_size))

                too_close = False
                for dx in range(-search_rad, search_rad + 1):
                    for dy in range(-search_rad, search_rad + 1):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < grid_w and 0 <= ny < grid_h:
                            for ax, ay in occ_grid[nx][ny]:
                                if np.hypot(ax - cx, ay - cy) < h_loc * 0.85:
                                    too_close = True
                                    break
                        if too_close: break
                    if too_close: break

                if not too_close:
                    i_nodes.append([cx, cy])
                    occ_grid[gx][gy].append([cx, cy])

            i_nodes = np.array(i_nodes)
            if len(i_nodes) > 0:
                self.fem_nodes = np.vstack((b_nodes, i_nodes))
            else:
                self.fem_nodes = b_nodes
        else:
            self.fem_nodes = b_nodes

        self.check_abort()
        tri = Delaunay(self.fem_nodes)

        def tri_in_solid(p1, p2, p3):
            if self.is_solid(*(p1 + p2 + p3) / 3.0): return True
            if self.is_solid(*(p1 + p2) / 2.0): return True
            if self.is_solid(*(p2 + p3) / 2.0): return True
            if self.is_solid(*(p3 + p1) / 2.0): return True
            if self.is_solid(*(p1 * 0.25 + p2 * 0.75)): return True
            if self.is_solid(*(p1 * 0.75 + p2 * 0.25)): return True
            if self.is_solid(*(p2 * 0.25 + p3 * 0.75)): return True
            if self.is_solid(*(p2 * 0.75 + p3 * 0.25)): return True
            if self.is_solid(*(p3 * 0.25 + p1 * 0.75)): return True
            if self.is_solid(*(p3 * 0.75 + p1 * 0.25)): return True
            return False

        valid_elements = []
        for idx, simplex in enumerate(tri.simplices):
            if idx % 10000 == 0: self.check_abort()
            pts = self.fem_nodes[simplex]
            if not tri_in_solid(pts[0], pts[1], pts[2]):
                valid_elements.append(simplex)

        self.fem_elements = np.array(valid_elements)

    def solve_filtration_fem(self, Hu, Hd):
        N_nodes = len(self.fem_nodes)
        I, J, V = [], [], []

        for idx, elem in enumerate(self.fem_elements):
            if idx % 10000 == 0: self.check_abort()
            pts = self.fem_nodes[elem]
            x, y = pts[:, 0], pts[:, 1]
            A = 0.5 * abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))
            if A == 0: continue

            b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
            c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]

            Ke = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    Ke[i, j] = (b[i] * b[j] + c[i] * c[j]) / (4 * A) * self.k

            for i in range(3):
                for j in range(3):
                    I.append(elem[i])
                    J.append(elem[j])
                    V.append(Ke[i, j])

        self.check_abort()
        K_global = sp.coo_matrix((V, (I, J)), shape=(N_nodes, N_nodes)).tolil()
        F_global = np.zeros(N_nodes)

        boundary_nodes = []
        for i, (x, y) in enumerate(self.fem_nodes):
            if y <= 0.01:
                if x <= self.dam_start_m:
                    boundary_nodes.append((i, Hu))
                elif x >= self.dam_end_m:
                    boundary_nodes.append((i, Hd))

        for idx, val in boundary_nodes:
            K_global[idx, :] = 0
            K_global[idx, idx] = 1.0
            F_global[idx] = val

        diag = K_global.diagonal()
        isolated_nodes = np.where(diag == 0)[0]
        for idx in isolated_nodes:
            K_global[idx, idx] = 1.0
            F_global[idx] = 0.0

        self.check_abort()
        K_global = K_global.tocsc()
        self.fem_H = spla.spsolve(K_global, F_global)

        self.check_abort()
        grid_x, grid_y = np.meshgrid(np.linspace(0, self.Lx, self.Nx), np.linspace(0, self.Ly, self.Ny))

        interp_H_lin = griddata(self.fem_nodes, self.fem_H, (grid_x, grid_y), method='linear')
        self.check_abort()
        interp_H_near = griddata(self.fem_nodes, self.fem_H, (grid_x, grid_y), method='nearest')
        self.check_abort()

        interp_H = np.where(np.isnan(interp_H_lin), interp_H_near, interp_H_lin)

        fluid_mask = (self.grid == 1)
        self.H[fluid_mask] = interp_H[fluid_mask]

    def solve_filtration_fdm(self, Hu, Hd, max_iter=200000):
        for i in range(self.Nx):
            if i < self.dam_start_idx:
                val = Hu
            elif i > self.dam_end_idx:
                val = Hd
            else:
                val = Hu - (Hu - Hd) * ((i - self.dam_start_idx) / max(1, self.dam_end_idx - self.dam_start_idx))
            self.H[:, i] = val

        mask = (self.grid == 1)
        for idx in range(max_iter):
            if idx % 100 == 0: self.check_abort()
            H_old = self.H.copy()
            H_C = self.H[1:-1, 1:-1]
            H_L = np.where(self.grid[1:-1, 0:-2] == 1, self.H[1:-1, 0:-2], H_C)
            H_R = np.where(self.grid[1:-1, 2:] == 1, self.H[1:-1, 2:], H_C)
            H_U = np.where(self.grid[0:-2, 1:-1] == 1, self.H[0:-2, 1:-1], H_C)
            H_D = np.where(self.grid[2:, 1:-1] == 1, self.H[2:, 1:-1], H_C)

            self.H[1:-1, 1:-1][mask[1:-1, 1:-1]] = 0.25 * (H_L + H_R + H_U + H_D)[mask[1:-1, 1:-1]]

            self.H[0, :self.dam_start_idx] = Hu
            self.H[0, self.dam_end_idx:] = Hd
            self.H[-1, :] = self.H[-2, :]
            self.H[:, 0] = self.H[:, 1]
            self.H[:, -1] = self.H[:, -2]

            if np.max(np.abs(self.H - H_old)) < 1e-4: break

    def calculate_velocities(self):
        H_R, H_L = np.roll(self.H, -1, axis=1), np.roll(self.H, 1, axis=1)
        H_D, H_U = np.roll(self.H, -1, axis=0), np.roll(self.H, 1, axis=0)

        G_R, G_L = np.roll(self.grid, -1, axis=1), np.roll(self.grid, 1, axis=1)
        G_D, G_U = np.roll(self.grid, -1, axis=0), np.roll(self.grid, 1, axis=0)

        G_R[:, -1] = 0
        G_L[:, 0] = 0
        G_D[-1, :] = 0
        G_U[0, :] = 0

        H_next_x = np.where(G_R == 1, H_R, self.H)
        H_prev_x = np.where(G_L == 1, H_L, self.H)
        dist_x = np.where(G_R == 1, self.dx, 0) + np.where(G_L == 1, self.dx, 0)
        gx = np.where(dist_x > 0, (H_next_x - H_prev_x) / dist_x, 0)

        H_next_y = np.where(G_D == 1, H_D, self.H)
        H_prev_y = np.where(G_U == 1, H_U, self.H)
        dist_y = np.where(G_D == 1, self.dy, 0) + np.where(G_U == 1, self.dy, 0)
        gy = np.where(dist_y > 0, (H_next_y - H_prev_y) / dist_y, 0)

        self.Vx, self.Vy = -self.k * gx, -self.k * gy
        self.Vx[self.grid == 0], self.Vy[self.grid == 0] = 0, 0

    def pollution_simulation(self, days, s_start, s_end, s_conc, discharge_duration=float('inf'), method="FDM"):
        if method == "FEM":
            N_nodes = len(self.fem_nodes)
            I_m, J_m, V_m = [], [], []
            I_k, J_k, V_k = [], [], []

            max_v_darcy = 1e-10

            for idx, elem in enumerate(self.fem_elements):
                if idx % 10000 == 0: self.check_abort()
                pts = self.fem_nodes[elem]
                x, y = pts[:, 0], pts[:, 1]

                det = x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
                A = abs(det) / 2.0
                if A < 1e-10: continue

                sign = np.sign(det)
                b = sign * np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
                c = sign * np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

                H_e = self.fem_H[elem]
                Vx_e = -self.k * sum(H_e[i] * b[i] for i in range(3)) / (2.0 * A)
                Vy_e = -self.k * sum(H_e[i] * c[i] for i in range(3)) / (2.0 * A)

                u = Vx_e / self.n
                v = Vy_e / self.n
                v_mag = np.hypot(u, v)
                max_v_darcy = max(max_v_darcy, abs(Vx_e) + abs(Vy_e))

                h_e = np.sqrt(2.0 * A)

                if v_mag > 1e-10:
                    Pe = (v_mag * h_e) / (2.0 * self.D + 1e-12)
                    tau = (h_e / (2.0 * v_mag)) * max(0.0, 1.0 - 1.0 / Pe)
                else:
                    tau = 0.0

                Me = (A / 12.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])

                for i in range(3):
                    for j in range(3):
                        term_i = u * b[i] + v * c[i]
                        term_j = u * b[j] + v * c[j]

                        K_diff = self.D * (b[i] * b[j] + c[i] * c[j]) / (4.0 * A)
                        K_conv = term_j / 6.0
                        K_supg = tau * term_i * term_j / (4.0 * A)

                        I_m.append(elem[i])
                        J_m.append(elem[j])
                        V_m.append(Me[i, j])

                        I_k.append(elem[i])
                        J_k.append(elem[j])
                        V_k.append(K_diff + K_conv + K_supg)

            self.check_abort()
            M_glob = sp.coo_matrix((V_m, (I_m, J_m)), shape=(N_nodes, N_nodes)).tocsr()
            K_glob = sp.coo_matrix((V_k, (I_k, J_k)), shape=(N_nodes, N_nodes)).tocsr()

            boundary_nodes = []
            source_nodes = set()

            for i, (nx, ny) in enumerate(self.fem_nodes):
                if ny <= 0.01 and nx <= self.dam_start_m:
                    boundary_nodes.append(i)
                    if s_start <= nx <= s_end:
                        source_nodes.add(i)

            if not source_nodes:
                surf = [i for i, (nx, ny) in enumerate(self.fem_nodes) if ny <= 0.01 and nx <= self.dam_start_m]
                if surf:
                    closest = min(surf, key=lambda i: abs(self.fem_nodes[i][0] - (s_start + s_end) / 2.0))
                    source_nodes.add(closest)

            limit = (
                    2 * self.D / self.dx ** 2 + 2 * self.D / self.dy ** 2 + max_v_darcy / self.dx + max_v_darcy / self.dy)
            dt = 0.8 * (1.0 / limit) * self.n
            steps = int(days / dt)

            self.check_abort()
            A_sys = (M_glob + dt * K_glob).tolil()
            for idx in boundary_nodes:
                A_sys[idx, :] = 0
                A_sys[idx, idx] = 1.0

            diag = A_sys.diagonal()
            isolated_nodes = np.where(diag == 0)[0]
            for idx in isolated_nodes:
                A_sys[idx, idx] = 1.0

            self.check_abort()
            A_sys = A_sys.tocsc()
            solve_A = spla.factorized(A_sys)

            self.fem_C = np.zeros(N_nodes)
            self.fem_history = []

            last_day = -1
            for step in range(steps):
                current_time = step * dt
                if step % 50 == 0: self.check_abort()

                if int(current_time) > last_day:
                    self.fem_history.append(self.fem_C.copy())
                    last_day = int(current_time)

                RHS = M_glob.dot(self.fem_C)
                is_discharging = current_time <= discharge_duration

                for idx in boundary_nodes:
                    if idx in source_nodes and is_discharging:
                        RHS[idx] = s_conc
                    else:
                        RHS[idx] = 0.0

                for idx in isolated_nodes:
                    RHS[idx] = 0.0

                self.fem_C = solve_A(RHS)
                self.fem_C = np.clip(self.fem_C, 0, s_conc)

            self.fem_history.append(self.fem_C.copy())

        else:
            self.C.fill(0)
            self.history = []
            if self.fem_nodes is not None:
                self.fem_history = []

            max_v = np.max(np.abs(self.Vx)) + np.max(np.abs(self.Vy))
            if max_v == 0: max_v = 1e-5
            limit = (2 * self.D / self.dx ** 2 + 2 * self.D / self.dy ** 2 + max_v / self.dx + max_v / self.dy)
            dt = 0.8 * (1.0 / limit) * self.n
            steps = int(days / dt)

            idx_start, idx_end = int(s_start / self.dx), int(s_end / self.dx)
            if idx_start == idx_end: idx_end += 1
            s_idx = slice(idx_start, idx_end)

            vx, vy = self.Vx[1:-1, 1:-1], self.Vy[1:-1, 1:-1]
            mask = (self.grid[1:-1, 1:-1] == 1)

            if self.fem_nodes is not None:
                fluid_y, fluid_x = np.where(self.grid == 1)
                fluid_coords = np.column_stack((fluid_x * self.dx, fluid_y * self.dy))
                fluid_tree = cKDTree(fluid_coords)
                _, nearest_fluid_idx = fluid_tree.query(self.fem_nodes)
                node_x_idx = fluid_x[nearest_fluid_idx]
                node_y_idx = fluid_y[nearest_fluid_idx]

            last_day = -1
            for step in range(steps):
                current_time = step * dt
                if step % 50 == 0: self.check_abort()

                if int(current_time) > last_day:
                    self.history.append(self.C.copy())
                    if self.fem_nodes is not None:
                        self.fem_history.append(self.C[node_y_idx, node_x_idx])
                    last_day = int(current_time)

                C_C = self.C[1:-1, 1:-1]
                C_L = self.C[1:-1, :-2]
                C_R = self.C[1:-1, 2:]
                C_U = self.C[:-2, 1:-1]
                C_D = self.C[2:, 1:-1]

                dCdx = np.where(vx > 0, (C_C - C_L) / self.dx, (C_R - C_C) / self.dx)
                dCdy = np.where(vy > 0, (C_C - C_U) / self.dy, (C_D - C_C) / self.dy)
                conv = -(vx * dCdx + vy * dCdy)

                diff = self.D * ((C_R - 2 * C_C + C_L) / self.dx ** 2 + (C_D - 2 * C_C + C_U) / self.dy ** 2)

                self.C[1:-1, 1:-1][mask] += (dt / self.n) * (conv + self.n * diff)[mask]

                self.C[0, :self.dam_start_idx] = 0.0
                if current_time <= discharge_duration:
                    self.C[0, s_idx] = s_conc

                self.C[0, self.dam_end_idx:] = self.C[1, self.dam_end_idx:]
                self.C[-1, :] = self.C[-2, :]
                self.C[:, 0] = self.C[:, 1]
                self.C[:, -1] = self.C[:, -2]
                self.C[self.grid == 0] = 0.0
                self.C = np.clip(self.C, 0, s_conc)

            self.history.append(self.C.copy())
            if self.fem_nodes is not None:
                self.fem_history.append(self.C[node_y_idx, node_x_idx])


class ResultView:
    def __init__(self, parent_frame, app, sim, method, title, is_detached=False, calc_time=None):
        self.parent_frame = parent_frame
        self.app = app
        self.sim = sim
        self.method = method
        self.title = title
        self.calc_time = calc_time

        self.frame = ttk.Frame(parent_frame)
        self.frame.pack(fill=tk.BOTH, expand=True)

        if self.calc_time is not None:
            timer_lbl = ttk.Label(self.frame, text=f"{tr('Час розрахунку: ')}{self.calc_time:.3f}{tr(' с')}")
            timer_lbl.pack(side=tk.TOP, pady=5)

        self.show_mesh_var = tk.BooleanVar(value=True)

        toolbar = ttk.Frame(self.frame)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Checkbutton(toolbar, text=tr("Відображати сітку"), variable=self.show_mesh_var, command=self.redraw_plots).pack(
            side=tk.LEFT, padx=5, pady=5)

        if not is_detached:
            ttk.Button(toolbar, text=tr("Винести в окреме вікно"), command=self.detach).pack(side=tk.RIGHT, padx=5, pady=5)
            ttk.Button(toolbar, text=tr("Закрити"), command=self.close_tab).pack(side=tk.RIGHT, padx=5, pady=5)
        else:
            ttk.Button(toolbar, text=tr("Згорнути"), command=self.attach).pack(side=tk.RIGHT, padx=5, pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.25)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.slider_ax = self.fig.add_axes([0.25, 0.04, 0.5, 0.02])

        hist_len = len(self.sim.fem_history) if self.method == "FEM" else len(self.sim.history)
        self.valmax = max(0, hist_len - 1)
        self.slider = Slider(self.slider_ax, tr('День'), 0, self.valmax, valinit=0, valstep=1)
        self.slider.on_changed(self.update_plot)

        self.X, self.Y = np.meshgrid(np.linspace(0, self.sim.Lx, self.sim.Nx),
                                     np.linspace(0, self.sim.Ly, self.sim.Ny))

        self.draw_static()

    def attach(self):
        current_val = self.slider.val
        current_mesh = self.show_mesh_var.get()

        top = self.parent_frame.winfo_toplevel()
        plt.close(self.fig)
        if self in self.app.result_views:
            self.app.result_views.remove(self)
        top.destroy()

        tab_frame = ttk.Frame(self.app.notebook)
        self.app.notebook.add(tab_frame, text=self.title)

        new_view = ResultView(tab_frame, self.app, self.sim, self.method, self.title, is_detached=False,
                              calc_time=self.calc_time)
        new_view.show_mesh_var.set(current_mesh)
        new_view.slider.set_val(current_val)

        self.app.result_views.append(new_view)
        self.app.notebook.select(tab_frame)

    def detach(self):
        current_val = self.slider.val
        current_mesh = self.show_mesh_var.get()

        if self.app.notebook:
            try:
                self.app.notebook.forget(self.parent_frame)
            except tk.TclError:
                pass

        plt.close(self.fig)
        self.parent_frame.destroy()
        if self in self.app.result_views:
            self.app.result_views.remove(self)

        top = tk.Toplevel(self.app.root)
        top.title(self.title)
        top.geometry("900x800")

        new_view = ResultView(top, self.app, self.sim, self.method, self.title, is_detached=True,
                              calc_time=self.calc_time)
        new_view.show_mesh_var.set(current_mesh)
        new_view.slider.set_val(current_val)
        self.app.result_views.append(new_view)

        def on_close():
            plt.close(new_view.fig)
            if new_view in self.app.result_views:
                self.app.result_views.remove(new_view)
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_close)

    def close_tab(self):
        if self.app.notebook:
            try:
                self.app.notebook.forget(self.parent_frame)
            except tk.TclError:
                pass
        plt.close(self.fig)
        self.parent_frame.destroy()
        if self in self.app.result_views:
            self.app.result_views.remove(self)

    def redraw_plots(self):
        has_history = (hasattr(self.sim, 'history') and self.sim.history) or \
                      (hasattr(self.sim, 'fem_history') and self.sim.fem_history)
        if has_history:
            self.draw_static()
            self.canvas.draw_idle()

    def draw_solid_geometry(self, ax):
        thick = 0.5
        vertices = []

        vertices.append((self.sim.dam_start_m, 0))
        vertices.append((self.sim.dam_start_m, self.sim.dam_depth_m))

        sorted_piles = sorted(self.sim.p_list_data, key=lambda p: p[1])

        for _, px, plen in sorted_piles:
            vertices.append((px, self.sim.dam_depth_m))
            vertices.append((px, self.sim.dam_depth_m + plen))
            vertices.append((px + thick, self.sim.dam_depth_m + plen))
            vertices.append((px + thick, self.sim.dam_depth_m))

        vertices.append((self.sim.dam_end_m, self.sim.dam_depth_m))
        vertices.append((self.sim.dam_end_m, 0))

        poly = patches.Polygon(vertices, closed=True, linewidth=0.5, edgecolor='black', facecolor='white', zorder=10)
        ax.add_patch(poly)

    def draw_static(self):
        self.ax1.clear()
        self.ax1.set_xlim(0, self.sim.Lx)
        self.ax1.set_ylim(self.sim.Ly, 0)
        self.ax1.set_aspect('auto')

        if self.method == "FEM" and self.sim.fem_elements is not None:
            self.ax1.tricontourf(self.sim.fem_nodes[:, 0], self.sim.fem_nodes[:, 1], self.sim.fem_elements,
                                 self.sim.fem_H, 20, cmap='Blues', alpha=0.5, antialiased=False)
            if self.show_mesh_var.get():
                self.ax1.triplot(self.sim.fem_nodes[:, 0], self.sim.fem_nodes[:, 1], self.sim.fem_elements,
                                 color='black', lw=0.3, alpha=0.3)
        elif self.method == "FDM":
            if self.show_mesh_var.get():
                x_lines = np.arange(0, self.sim.Lx + 1e-5, self.sim.dx)
                y_lines = np.arange(0, self.sim.Ly + 1e-5, self.sim.dy)
                self.ax1.vlines(x_lines, 0, self.sim.Ly, color='black', lw=0.2, alpha=0.3)
                self.ax1.hlines(y_lines, 0, self.sim.Lx, color='black', lw=0.2, alpha=0.3)

            H_plot = self.sim.H.copy()
            for _ in range(3): H_plot = np.where(self.sim.grid == 0, np.maximum.reduce(
                [H_plot, np.roll(H_plot, 1, axis=0), np.roll(H_plot, -1, axis=0), np.roll(H_plot, 1, axis=1),
                 np.roll(H_plot, -1, axis=1)]), H_plot)

            self.ax1.contourf(self.X, self.Y, H_plot, 20, cmap='Blues', alpha=0.5)

        v_plot = np.where(self.sim.grid == 0, np.nan, self.sim.Vx)
        self.ax1.streamplot(self.X, self.Y, v_plot, np.where(self.sim.grid == 0, np.nan, self.sim.Vy), color='k',
                            linewidth=0.8, density=1.5)

        self.draw_solid_geometry(self.ax1)
        self.ax1.set_title(tr("Напір"))
        self.update_plot(self.slider.val)

    def update_plot(self, val):
        day = int(val)
        self.ax2.clear()
        self.ax2.set_xlim(0, self.sim.Lx)
        self.ax2.set_ylim(self.sim.Ly, 0)
        self.ax2.set_aspect('auto')

        if self.method == "FEM" and self.sim.fem_elements is not None:
            day_idx = min(day, len(self.sim.fem_history) - 1)
            self.ax2.tripcolor(self.sim.fem_nodes[:, 0], self.sim.fem_nodes[:, 1], self.sim.fem_elements,
                               self.sim.fem_history[day_idx], shading='gouraud', cmap='Reds', vmin=0, vmax=1)
            if self.show_mesh_var.get():
                self.ax2.triplot(self.sim.fem_nodes[:, 0], self.sim.fem_nodes[:, 1], self.sim.fem_elements,
                                 color='black', lw=0.3, alpha=0.3)
        else:
            if self.method == "FDM":
                if self.show_mesh_var.get():
                    x_lines = np.arange(0, self.sim.Lx + 1e-5, self.sim.dx)
                    y_lines = np.arange(0, self.sim.Ly + 1e-5, self.sim.dy)
                    self.ax2.vlines(x_lines, 0, self.sim.Ly, color='black', lw=0.2, alpha=0.3)
                    self.ax2.hlines(y_lines, 0, self.sim.Lx, color='black', lw=0.2, alpha=0.3)

            if hasattr(self, 'sim') and self.sim.history:
                day_idx = min(day, len(self.sim.history) - 1)

                C_plot = self.sim.history[day_idx].copy()
                self.ax2.contourf(self.X, self.Y, C_plot, 40, cmap='Reds', vmin=0, vmax=1)

        self.draw_solid_geometry(self.ax2)
        self.ax2.set_title(f"{tr('Забруднення: День ')}{day}")
        self.canvas.draw_idle()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(tr("Конструктор гребель"))
        self.abort_request = False

        self.calc_counter = 0
        self.result_views = []

        side_panel = ttk.Frame(root, padding="10")
        side_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        lang_frame = ttk.Frame(side_panel)
        lang_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(lang_frame, text="UA", command=lambda: self.change_lang("ua")).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(lang_frame, text="EN", command=lambda: self.change_lang("en")).pack(side=tk.LEFT, expand=True, fill=tk.X)

        method_f = ttk.LabelFrame(side_panel, text=tr(" Метод розрахунку "), padding="10")
        method_f.pack(fill=tk.X, pady=5)
        self.method_var = tk.StringVar(value="FEM")
        ttk.Radiobutton(method_f, text=tr("Скінченні елементи"), variable=self.method_var, value="FEM").pack(
            anchor=tk.W)
        ttk.Radiobutton(method_f, text=tr("Скінченні різниці"), variable=self.method_var, value="FDM").pack(
            anchor=tk.W)

        geom_domain_f = ttk.LabelFrame(side_panel, text=tr(" Геометрія області "), padding="10")
        geom_domain_f.pack(fill=tk.X, pady=5)
        geom_domain_f.columnconfigure(0, weight=1)
        self.geom_vars = {
            'Ширина області (м)': tk.DoubleVar(value=50),
            'Глибина області (м)': tk.DoubleVar(value=25)
        }
        for i, (k, v) in enumerate(self.geom_vars.items()):
            ttk.Label(geom_domain_f, text=tr(k)).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Entry(geom_domain_f, textvariable=v, width=8, justify='right').grid(row=i, column=1, sticky=tk.E)

        geom_dam_f = ttk.LabelFrame(side_panel, text=tr(" Геометрія греблі "), padding="10")
        geom_dam_f.pack(fill=tk.X, pady=5)
        geom_dam_f.columnconfigure(0, weight=1)
        self.dam_vars = {
            'Початок греблі (м)': tk.DoubleVar(value=15),
            'Кінець греблі (м)': tk.DoubleVar(value=35),
            'Глибина основи (м)': tk.DoubleVar(value=5)
        }
        for i, (k, v) in enumerate(self.dam_vars.items()):
            ttk.Label(geom_dam_f, text=tr(k)).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Entry(geom_dam_f, textvariable=v, width=8, justify='right').grid(row=i, column=1, sticky=tk.E)

        self.piles_f = ttk.LabelFrame(side_panel, text=tr(" Шпунти "), padding="10")
        self.piles_f.pack(fill=tk.X, pady=5)
        self.p_frame = ttk.Frame(self.piles_f)
        self.p_frame.pack(fill=tk.X)
        self.p_list = []
        ttk.Button(self.piles_f, text=tr("+ Додати шпунт"), command=self.add_p_ui).pack(fill=tk.X, pady=(5, 0))

        phys_f = ttk.LabelFrame(side_panel, text=tr(" Фізичні властивості "), padding="10")
        phys_f.pack(fill=tk.X, pady=5)
        phys_f.columnconfigure(0, weight=1)
        self.phys_vars = {
            'Коеф. фільтрації': tk.DoubleVar(value=2.0),
            'Коеф. дифузії': tk.DoubleVar(value=0.01),
            'Пористість': tk.DoubleVar(value=0.3),
            "Напір лівого б'єфу (м)": tk.DoubleVar(value=12),
            "Напір правого б'єфу (м)": tk.DoubleVar(value=4),
            'Період (діб)': tk.IntVar(value=45)
        }
        for i, (k, v) in enumerate(self.phys_vars.items()):
            ttk.Label(phys_f, text=tr(k)).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Entry(phys_f, textvariable=v, width=8, justify='right').grid(row=i, column=1, sticky=tk.E)

        poll_f = ttk.LabelFrame(side_panel, text=tr(" Властивості забруднення "), padding="10")
        poll_f.pack(fill=tk.X, pady=5)
        poll_f.columnconfigure(0, weight=1)
        self.poll_vars = {
            'Початок (м)': tk.DoubleVar(value=8),
            'Кінець (м)': tk.DoubleVar(value=12),
            'Концентрація (0-1)': tk.DoubleVar(value=1.0)
        }
        for i, (k, v) in enumerate(self.poll_vars.items()):
            ttk.Label(poll_f, text=tr(k)).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Entry(poll_f, textvariable=v, width=8, justify='right').grid(row=i, column=1, sticky=tk.E)

        r = len(self.poll_vars)
        self.poll_type_var = tk.StringVar(value="continuous")
        ttk.Radiobutton(poll_f, text=tr("Постійне джерело"), variable=self.poll_type_var, value="continuous",
                        command=self.toggle_poll_duration).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        ttk.Radiobutton(poll_f, text=tr("Тимчасове джерело"), variable=self.poll_type_var, value="temporary",
                        command=self.toggle_poll_duration).grid(row=r + 1, column=0, columnspan=2, sticky=tk.W)

        self.poll_duration_lbl = ttk.Label(poll_f, text=tr("Тривалість забруднення (діб):"))
        self.poll_duration_lbl.grid(row=r + 2, column=0, sticky=tk.W, pady=2)

        self.poll_duration_var = tk.DoubleVar(value=10.0)
        self.poll_duration_entry = ttk.Entry(poll_f, textvariable=self.poll_duration_var, width=8, justify='right')
        self.poll_duration_entry.grid(row=r + 2, column=1, sticky=tk.E)

        self.toggle_poll_duration()

        self.btn_frame = ttk.Frame(side_panel)
        self.btn_frame.pack(fill=tk.X, pady=5)

        self.calc_btn = ttk.Button(self.btn_frame, text=tr("РОЗРАХУВАТИ"), command=self.start_calculation)
        self.calc_btn.pack(fill=tk.X)

        self.stop_btn = ttk.Button(self.btn_frame, text=tr("ЗУПИНИТИ РОЗРАХУНОК"), command=self.stop_calculation)

        self.reset_btn = ttk.Button(self.btn_frame, text=tr("Скинути налаштування"), command=self.reset_defaults)
        self.reset_btn.pack(fill=tk.X, pady=(5, 0))

        self.status_text = tk.Text(side_panel, height=3, width=30, wrap=tk.WORD, bg=root.cget("bg"), relief="flat",
                                   font=("TkDefaultFont", 9))
        self.status_text.pack(fill=tk.X, pady=2)
        self.set_status(tr("Готовий"), "gray")

        main_f = ttk.Frame(root)
        main_f.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_f)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.add_p_ui(20, 8)
        self.add_p_ui(30, 12)

    def change_lang(self, lang):
        set_lang(lang)
        for widget in self.root.winfo_children():
            widget.destroy()
        self.__init__(self.root)

    def toggle_poll_duration(self):
        if self.poll_type_var.get() == "temporary":
            self.poll_duration_lbl.grid()
            self.poll_duration_entry.grid()
        else:
            self.poll_duration_lbl.grid_remove()
            self.poll_duration_entry.grid_remove()

    def set_status(self, message, color="black"):
        self.status_text.config(state=tk.NORMAL, fg=color)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, message)
        self.status_text.config(state=tk.DISABLED)

    def add_p_ui(self, x=25, l=5):
        f = ttk.Frame(self.p_frame)
        f.pack(fill=tk.X, pady=2)
        xv, lv = tk.DoubleVar(value=x), tk.DoubleVar(value=l)

        ttk.Label(f, text=tr("Розташування:")).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=xv, width=5).pack(side=tk.LEFT, padx=(2, 5))

        ttk.Label(f, text=tr("Довжина:")).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=lv, width=5).pack(side=tk.LEFT, padx=(2, 5))

        d = (f, xv, lv)
        ttk.Button(f, text="X", width=2, command=lambda: [f.destroy(), self.p_list.remove(d)]).pack(side=tk.RIGHT)
        self.p_list.append(d)

    def reset_defaults(self):
        self.method_var.set("FEM")
        self.geom_vars['Ширина області (м)'].set(50)
        self.geom_vars['Глибина області (м)'].set(25)

        self.dam_vars['Початок греблі (м)'].set(15)
        self.dam_vars['Кінець греблі (м)'].set(35)
        self.dam_vars['Глибина основи (м)'].set(5)

        self.phys_vars['Коеф. фільтрації'].set(2.0)
        self.phys_vars['Коеф. дифузії'].set(0.01)
        self.phys_vars['Пористість'].set(0.3)
        self.phys_vars["Напір лівого б'єфу (м)"].set(12)
        self.phys_vars["Напір правого б'єфу (м)"].set(4)
        self.phys_vars['Період (діб)'].set(45)

        self.poll_vars['Початок (м)'].set(8)
        self.poll_vars['Кінець (м)'].set(12)
        self.poll_vars['Концентрація (0-1)'].set(1.0)

        self.poll_type_var.set("continuous")
        self.poll_duration_var.set(10.0)
        self.toggle_poll_duration()

        for f, _, _ in self.p_list:
            f.destroy()
        self.p_list.clear()
        self.add_p_ui(20, 8)
        self.add_p_ui(30, 12)
        self.set_status(tr("Налаштування скинуто"), "black")

    def start_calculation(self):
        self.abort_request = False
        self.calc_btn.config(state='disabled')

        self.stop_btn.pack(fill=tk.X, pady=(5, 0))
        self.stop_btn.config(state='normal')

        self.start_time = time.time()
        self.set_status(tr("Йде розрахунок..."), "red")
        threading.Thread(target=self.perform_task).start()

    def stop_calculation(self):
        self.abort_request = True
        self.stop_btn.config(state='disabled')
        self.set_status(tr("Зупиняємо..."), "orange")

    def reset_buttons(self):
        self.calc_btn.config(state='normal')
        self.stop_btn.pack_forget()

    def perform_task(self):
        try:
            gm = {k: v.get() for k, v in self.geom_vars.items()}
            dm = {k: v.get() for k, v in self.dam_vars.items()}
            ph = {k: v.get() for k, v in self.phys_vars.items()}
            pl = {k: v.get() for k, v in self.poll_vars.items()}

            self.sim = DamSimulation(gm['Ширина області (м)'], gm['Глибина області (м)'],
                                     ph['Коеф. фільтрації'], ph['Пористість'], ph['Коеф. дифузії'],
                                     abort_check=lambda: self.abort_request)

            self.sim.dam_start_m = dm['Початок греблі (м)']
            self.sim.dam_end_m = dm['Кінець греблі (м)']
            self.sim.dam_depth_m = dm['Глибина основи (м)']
            self.sim.p_list_data = [(None, xv.get(), lv.get()) for _, xv, lv in self.p_list]

            pts_x = [0.0, self.sim.Lx, self.sim.dam_start_m, self.sim.dam_end_m]
            pts_y = [0.0, self.sim.Ly, self.sim.dam_depth_m]
            for _, px, plen in self.sim.p_list_data:
                pts_x.extend([px, px + 0.5])
                pts_y.append(self.sim.dam_depth_m + plen)

            def get_step(vals):
                for st in [0.5, 0.25, 0.2, 0.1, 0.05]:
                    if all(abs(v / st - round(v / st)) < 1e-4 for v in vals): return st
                return 0.1

            self.sim.dx = get_step(pts_x)
            self.sim.dy = get_step(pts_y)

            p_dur = float('inf') if self.poll_type_var.get() == "continuous" else self.poll_duration_var.get()

            if self.method_var.get() == "FEM":
                self.sim.init_grids()
                self.sim.set_dam_geometry(self.sim.dam_start_m, self.sim.dam_end_m, self.sim.dam_depth_m)
                for _, x, l in self.sim.p_list_data: self.sim.add_pile(x, l)

                self.sim.build_fem_mesh(1.5)
                self.sim.solve_filtration_fem(ph["Напір лівого б'єфу (м)"], ph["Напір правого б'єфу (м)"])
                self.sim.calculate_velocities()
                self.sim.pollution_simulation(ph['Період (діб)'], pl['Початок (м)'], pl['Кінець (м)'],
                                              pl['Концентрація (0-1)'], discharge_duration=p_dur,
                                              method="FEM")
            else:
                self.sim.init_grids()
                self.sim.set_dam_geometry(self.sim.dam_start_m, self.sim.dam_end_m, self.sim.dam_depth_m)
                for _, x, l in self.sim.p_list_data: self.sim.add_pile(x, l)

                self.sim.fem_nodes = None
                self.sim.fem_elements = None

                self.sim.solve_filtration_fdm(ph["Напір лівого б'єфу (м)"], ph["Напір правого б'єфу (м)"])
                self.sim.calculate_velocities()
                self.sim.pollution_simulation(ph['Період (діб)'], pl['Початок (м)'], pl['Кінець (м)'],
                                              pl['Концентрація (0-1)'], discharge_duration=p_dur,
                                              method="FDM")

            self.root.after(0, self.on_calculation_complete)

        except InterruptedError:
            self.root.after(0, self.on_calculation_aborted)
        except Exception as e:
            self.root.after(0, lambda: self.set_status(f"{tr('Помилка: ')}{str(e)}", "red"))
            self.root.after(0, self.reset_buttons)

    def on_calculation_aborted(self):
        self.set_status(tr("Розрахунок зупинено"), "orange")
        self.reset_buttons()

    def on_calculation_complete(self):
        calc_time = time.time() - self.start_time
        self.calc_counter += 1
        title = f"{tr('Рішення ')}{self.calc_counter} ({tr('МСЕ') if self.method_var.get() == 'FEM' else tr('МСР')})"

        self.set_status(tr("Готово"), "green")
        self.reset_buttons()

        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=title)
        view = ResultView(tab_frame, self, self.sim, self.method_var.get(), title, is_detached=False,
                          calc_time=calc_time)
        self.result_views.append(view)
        self.notebook.select(tab_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()