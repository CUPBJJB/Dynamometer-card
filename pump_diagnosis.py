#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽油井泵功图计算与诊断系统
基于波动方程（Gibbs方法）求解泵功图
使用傅里叶级数重构地面示功图并与原始数据对比
"""

import numpy as np
from numpy import fft
from scipy import interpolate
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
import json
import os

# 配置matplotlib中文字体 (按优先级排列)
_chinese_font = 'sans-serif'
for _font_name in ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']:
    if any(_font_name == f.name for f in fm.fontManager.ttflist):
        _chinese_font = _font_name
        break
matplotlib.rcParams['font.sans-serif'] = [_chinese_font, 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 核心计算模块
# ============================================================

class WellModel:
    """油井模型，包含井参数和示功图数据"""

    def __init__(self, name="Well", rod_length=1000.0, rod_diameter=0.022,
                 rod_density=7850.0, elastic_modulus=2.1e11,
                 pump_speed=6.0, damping_coeff=0.1,
                 data_points=None):
        self.name = name
        self.rod_length = rod_length          # 抽油杆长度 (m)
        self.rod_diameter = rod_diameter      # 抽油杆直径 (m)
        self.rod_density = rod_density        # 抽油杆密度 (kg/m³)
        self.elastic_modulus = elastic_modulus  # 弹性模量 (Pa)
        self.pump_speed = pump_speed          # 冲次 (min⁻¹)
        self.damping_coeff = damping_coeff    # 阻尼系数 (s⁻¹)
        self.data_points = data_points        # [(位移, 载荷), ...]
        self.rod_area = np.pi * rod_diameter**2 / 4  # 抽油杆截面积 (m²)
        self.wave_speed = np.sqrt(elastic_modulus / rod_density)  # 波速 (m/s)

    @property
    def period(self):
        """冲程周期 (s)"""
        return 60.0 / self.pump_speed

    @property
    def angular_freq(self):
        """角频率 (rad/s)"""
        return 2 * np.pi * self.pump_speed / 60.0

    def copy(self):
        return WellModel(
            name=self.name,
            rod_length=self.rod_length,
            rod_diameter=self.rod_diameter,
            rod_density=self.rod_density,
            elastic_modulus=self.elastic_modulus,
            pump_speed=self.pump_speed,
            damping_coeff=self.damping_coeff,
            data_points=list(self.data_points) if self.data_points else None
        )


def load_well_data(filepath):
    """从文本文件加载油井数据 (格式: 载荷 位移)"""
    disp = []
    load = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    load_val = float(parts[0])
                    disp_val = float(parts[1])
                    disp.append(disp_val)
                    load.append(load_val)
                except ValueError:
                    continue
    return list(zip(disp, load))


def fourier_fit(data_y, n_harmonics=10):
    """
    对周期性数据做傅里叶级数拟合
    返回傅里叶系数: a0, a_n, b_n (n=1..n_harmonics)
    f(θ) ≈ a0/2 + Σ[a_n cos(nθ) + b_n sin(nθ)]
    """
    N = len(data_y)
    y = np.array(data_y)
    fft_coeff = fft.fft(y) / N

    a0 = 2 * np.real(fft_coeff[0])
    a_n = []
    b_n = []
    for n in range(1, n_harmonics + 1):
        if n < N:
            a_n.append(2 * np.real(fft_coeff[n]))
            b_n.append(-2 * np.imag(fft_coeff[n]))
        else:
            a_n.append(0.0)
            b_n.append(0.0)

    return a0, np.array(a_n), np.array(b_n)


def fourier_reconstruct(theta, a0, a_n, b_n):
    """从傅里叶系数重构信号"""
    result = a0 / 2.0
    for n in range(len(a_n)):
        result += a_n[n] * np.cos((n + 1) * theta) + b_n[n] * np.sin((n + 1) * theta)
    return result


def solve_wave_equation_fourier(well, n_harmonics=10, n_pts=200):
    """
    使用傅里叶级数法（Gibbs方法）求解波动方程
    从地面示功图计算泵功图

    波动方程: ∂²u/∂t² = a² ∂²u/∂x² - c ∂u/∂t
    a = √(E/ρ) 为波速, c 为阻尼系数

    边界条件:
      地面 (x=0): 位移 u(0,t) = D(t), 载荷 F(0,t) = EA ∂u/∂x|₀
      泵 (x=L):   位移 u(L,t), 载荷 F(L,t) = EA ∂u/∂x|_L
    """
    L = well.rod_length
    a = well.wave_speed
    c = well.damping_coeff
    EA = well.elastic_modulus * well.rod_area
    omega = well.angular_freq

    # 原始数据
    raw_disp = np.array([p[0] for p in well.data_points])
    raw_load = np.array([p[1] for p in well.data_points])
    N_raw = len(raw_disp)

    # 生成均匀角度网格 (0 到 2π)
    theta_raw = np.linspace(0, 2 * np.pi, N_raw)
    theta_new = np.linspace(0, 2 * np.pi, n_pts)

    # 插值到均匀网格
    disp_interp = interpolate.interp1d(theta_raw, raw_disp, kind='cubic')
    load_interp = interpolate.interp1d(theta_raw, raw_load, kind='cubic')
    disp = disp_interp(theta_new)
    force = load_interp(theta_new)

    # 傅里叶拟合地面位移和载荷
    # D(θ) = σ₀/2 + Σ[σ_n cos(nθ) + τ_n sin(nθ)]
    # F(θ) = ν₀/2 + Σ[ν_n cos(nθ) + δ_n sin(nθ)]
    sigma0, sigma_n, tau_n = fourier_fit(disp, n_harmonics)
    nu0, nu_n, delta_n = fourier_fit(force, n_harmonics)

    # 计算波数 (复波数法处理阻尼)
    # k_n = α_n + iβ_n
    alpha_n = np.zeros(n_harmonics)
    beta_n = np.zeros(n_harmonics)

    for n in range(1, n_harmonics + 1):
        n_omega = n * omega
        # √(1 + (c/(nω))²)
        sqrt_term = np.sqrt(1 + (c / n_omega) ** 2)
        alpha_n[n - 1] = (n_omega / (a * np.sqrt(2))) * np.sqrt(sqrt_term - 1)
        beta_n[n - 1] = (n_omega / (a * np.sqrt(2))) * np.sqrt(sqrt_term + 1)

    # 计算泵处的位移和载荷系数
    # 使用Gibbs递推公式
    pump_disp_coeff_a0 = sigma0 / 2  # a0项 (位移)
    pump_disp_coeff_cos = np.zeros(n_harmonics)  # cos系数
    pump_disp_coeff_sin = np.zeros(n_harmonics)  # sin系数

    pump_load_coeff_a0 = nu0 / 2  # a0项 (载荷)
    pump_load_coeff_cos = np.zeros(n_harmonics)
    pump_load_coeff_sin = np.zeros(n_harmonics)

    for n in range(1, n_harmonics + 1):
        idx = n - 1
        alpha = alpha_n[idx]
        beta = beta_n[idx]

        # 双曲/三角函数在深度L处的值
        ch = np.cosh(beta * L)
        sh = np.sinh(beta * L)
        ca = np.cos(alpha * L)
        sa = np.sin(alpha * L)

        # 位移传递系数
        # u_pump: O_n = σ_n·ch·ca + τ_n·sh·sa + (ν_n/(EA·k_n))·(...)
        # 简化处理 (忽略阻尼耦合项在位移中的直接贡献)
        disp_factor_real = ch * ca
        disp_factor_imag = sh * sa

        pump_disp_coeff_cos[idx] = sigma_n[idx] * disp_factor_real + tau_n[idx] * disp_factor_imag
        pump_disp_coeff_sin[idx] = tau_n[idx] * disp_factor_real - sigma_n[idx] * disp_factor_imag

        # 载荷传递系数
        load_factor_real = ch * ca
        load_factor_imag = -sh * sa

        pump_load_coeff_cos[idx] = nu_n[idx] * load_factor_real + delta_n[idx] * load_factor_imag
        pump_load_coeff_sin[idx] = delta_n[idx] * load_factor_real - nu_n[idx] * load_factor_imag

    # 重构泵功图
    pump_disp = np.zeros(n_pts)
    pump_load = np.zeros(n_pts)

    for i, th in enumerate(theta_new):
        pump_disp[i] = pump_disp_coeff_a0
        pump_load[i] = pump_load_coeff_a0
        for n in range(1, n_harmonics + 1):
            idx = n - 1
            pump_disp[i] += (pump_disp_coeff_cos[idx] * np.cos(n * th) +
                             pump_disp_coeff_sin[idx] * np.sin(n * th))
            pump_load[i] += (pump_load_coeff_cos[idx] * np.cos(n * th) +
                             pump_load_coeff_sin[idx] * np.sin(n * th))

    # 重构地面功图 (用于对比)
    surface_disp_recon = np.zeros(n_pts)
    surface_load_recon = np.zeros(n_pts)
    for i, th in enumerate(theta_new):
        surface_disp_recon[i] = fourier_reconstruct(th, sigma0, sigma_n, tau_n)
        surface_load_recon[i] = fourier_reconstruct(th, nu0, nu_n, delta_n)

    return {
        'theta': theta_new,
        'surface_disp_original': disp,
        'surface_load_original': force,
        'surface_disp_recon': surface_disp_recon,
        'surface_load_recon': surface_load_recon,
        'pump_disp': pump_disp,
        'pump_load': pump_load,
        'fourier_coeff': {
            'disp_a0': sigma0, 'disp_an': sigma_n, 'disp_bn': tau_n,
            'load_a0': nu0, 'load_an': nu_n, 'load_bn': delta_n,
        },
        'harmonics': n_harmonics,
    }


def solve_wave_equation_fd(well, nx=50, nt_per_cycle=200, n_cycles=5):
    """
    使用有限差分法求解波动方程
    从地面示功图计算泵功图
    模拟多个周期直至达到稳态

    波动方程: ∂²u/∂t² = a² ∂²u/∂x² - c ∂u/∂t
    """
    L = well.rod_length
    a = well.wave_speed
    c = well.damping_coeff
    EA = well.elastic_modulus * well.rod_area
    T = well.period

    dx = L / nx
    dt = T / nt_per_cycle

    # CFL条件检查
    cfl = a * dt / dx
    if cfl > 0.9:
        nt_per_cycle = int(T * a / (0.9 * dx)) + 1
        dt = T / nt_per_cycle
        cfl = a * dt / dx

    total_steps = nt_per_cycle * n_cycles

    # 原始数据 - 生成多周期表面边界条件
    raw_disp = np.array([p[0] for p in well.data_points])
    raw_load = np.array([p[1] for p in well.data_points])
    N_raw = len(raw_disp)

    theta_raw = np.linspace(0, 2 * np.pi, N_raw, endpoint=False)
    theta_for_interp = np.append(theta_raw, 2 * np.pi)
    disp_for_interp = np.append(raw_disp, raw_disp[0])
    load_for_interp = np.append(raw_load, raw_load[0])

    disp_func = interpolate.interp1d(theta_for_interp, disp_for_interp, kind='cubic')
    load_func = interpolate.interp1d(theta_for_interp, load_for_interp, kind='cubic')

    # 生成全部时间步的地面边界条件
    theta_all = np.linspace(0, 2 * np.pi * n_cycles, total_steps + 1, endpoint=False)
    theta_mod = theta_all % (2 * np.pi)
    surface_disp_all = disp_func(theta_mod)
    surface_load_all = load_func(theta_mod)

    # 初始化: 静态位移分布 (基于地面载荷)
    u = np.zeros((nx + 1, total_steps + 1))
    u[0, 0] = surface_disp_all[0]
    u[1, 0] = u[0, 0] + surface_load_all[0] * dx / EA
    for i in range(2, nx + 1):
        u[i, 0] = u[i-1, 0] + (u[1, 0] - u[0, 0])

    # 时间推进系数
    rx = (a * dt / dx) ** 2
    damp_factor = c * dt / 2

    for j in range(total_steps):
        # 表面位移边界条件
        u[0, j + 1] = surface_disp_all[j + 1]

        # 从表面载荷计算u[1]
        u[1, j + 1] = u[0, j + 1] + surface_load_all[j + 1] * dx / EA

        # 内部节点
        for i in range(1, nx):
            if j == 0:
                # 第一步使用显式Euler
                u[i, j + 1] = u[i, j] + 0.5 * rx * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
            else:
                u_xx = u[i+1, j] - 2*u[i, j] + u[i-1, j]
                u[i, j + 1] = (2*u[i, j] - u[i, j-1] + rx * u_xx +
                               damp_factor * u[i, j-1]) / (1 + damp_factor)

        # 泵端边界条件 (自由端: 零应力)
        # 使用二阶精度外推: ∂u/∂x = 0 at x=L
        u[nx, j + 1] = (4*u[nx-1, j+1] - u[nx-2, j+1]) / 3

    # 提取最后一个周期的泵功图
    start = (n_cycles - 1) * nt_per_cycle
    end = total_steps + 1
    theta_out = np.linspace(0, 2 * np.pi, nt_per_cycle + 1, endpoint=False)

    pump_disp = u[nx, start:end]
    pump_load = EA * (u[nx, start:end] - u[nx-1, start:end]) / dx

    # 地面数据 (最后一个周期)
    surface_disp = surface_disp_all[start:end]
    surface_load = surface_load_all[start:end]

    return {
        'theta': theta_out,
        'surface_disp_original': surface_disp,
        'surface_load_original': surface_load,
        'pump_disp': pump_disp,
        'pump_load': pump_load,
        'nx': nx, 'nt': nt_per_cycle,
        'n_cycles': n_cycles,
    }


def compute_surface_reconstruction(well, n_harmonics=10, n_pts=200):
    """
    仅使用傅里叶级数重构地面示功图（用于对比）
    """
    raw_disp = np.array([p[0] for p in well.data_points])
    raw_load = np.array([p[1] for p in well.data_points])
    N_raw = len(raw_disp)

    theta_raw = np.linspace(0, 2 * np.pi, N_raw)
    theta_new = np.linspace(0, 2 * np.pi, n_pts)

    disp_interp = interpolate.interp1d(theta_raw, raw_disp, kind='cubic')
    load_interp = interpolate.interp1d(theta_raw, raw_load, kind='cubic')
    disp = disp_interp(theta_new)
    force = load_interp(theta_new)

    sigma0, sigma_n, tau_n = fourier_fit(disp, n_harmonics)
    nu0, nu_n, delta_n = fourier_fit(force, n_harmonics)

    disp_recon = np.array([fourier_reconstruct(th, sigma0, sigma_n, tau_n)
                           for th in theta_new])
    load_recon = np.array([fourier_reconstruct(th, nu0, nu_n, delta_n)
                           for th in theta_new])

    return {
        'theta': theta_new,
        'disp_original': disp,
        'load_original': force,
        'disp_recon': disp_recon,
        'load_recon': load_recon,
        'fourier_coeff': {
            'disp_a0': sigma0, 'disp_an': sigma_n, 'disp_bn': tau_n,
            'load_a0': nu0, 'load_an': nu_n, 'load_bn': delta_n,
        },
        'harmonics': n_harmonics,
    }


def export_results_csv(filepath, results):
    """导出计算结果为CSV"""
    data_cols = {}
    if 'surface_disp_original' in results:
        data_cols['Angle(rad)'] = results.get('theta', [])
        data_cols['Surface_Disp_Original(m)'] = results['surface_disp_original']
        data_cols['Surface_Load_Original(kN)'] = results['surface_load_original']

    if 'surface_disp_recon' in results:
        data_cols['Surface_Disp_Reconstructed(m)'] = results['surface_disp_recon']
        data_cols['Surface_Load_Reconstructed(kN)'] = results['surface_load_recon']

    if 'pump_disp' in results:
        data_cols['Pump_Disp(m)'] = results['pump_disp']
        data_cols['Pump_Load(kN)'] = results['pump_load']

    n_rows = max(len(v) for v in data_cols.values())
    header = list(data_cols.keys())

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for i in range(n_rows):
            row = []
            for key in header:
                val = data_cols[key][i] if i < len(data_cols[key]) else ''
                row.append(str(val))
            f.write(','.join(row) + '\n')


def export_results_json(filepath, results):
    """导出计算结果为JSON"""
    export_dict = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            export_dict[key] = value.tolist()
        elif isinstance(value, dict):
            export_dict[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    export_dict[key][k] = v.tolist()
                else:
                    export_dict[key][k] = v
        else:
            export_dict[key] = value

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_dict, f, indent=2, ensure_ascii=False)


# ============================================================
# GUI 模块
# ============================================================

class PumpDiagnosisApp:
    """抽油井泵功图诊断系统 GUI"""

    DEFAULT_WELL_PARAMS = {
        'DK3': {'rod_length': 1200.0, 'rod_diameter': 0.022, 'pump_speed': 6.0},
        'DK6': {'rod_length': 1500.0, 'rod_diameter': 0.025, 'pump_speed': 5.0},
        'S22': {'rod_length': 1000.0, 'rod_diameter': 0.022, 'pump_speed': 7.0},
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("抽油井泵功图计算与诊断系统")
        self.root.geometry("1400x900")

        self.well = None
        self.surface_results = None
        self.pump_results_fourier = None
        self.pump_results_fd = None
        self.current_file = None

        self._setup_ui()
        self._load_default_data()

    def _setup_ui(self):
        """设置用户界面"""
        # 顶部控制面板
        control_frame = ttk.Frame(self.root, padding=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # 文件操作
        file_frame = ttk.LabelFrame(control_frame, text="数据文件", padding=5)
        file_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Button(file_frame, text="导入数据...", command=self._import_data).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="导出结果...", command=self._export_results).pack(
            side=tk.LEFT, padx=2)

        # 快速选择油井
        well_frame = ttk.LabelFrame(control_frame, text="选择油井", padding=5)
        well_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.well_var = tk.StringVar(value='DK3')
        for name in ['DK3', 'DK6', 'S22']:
            ttk.Radiobutton(well_frame, text=name, variable=self.well_var,
                            value=name, command=self._on_well_changed).pack(
                side=tk.LEFT, padx=5)

        # 参数设置
        param_frame = ttk.LabelFrame(control_frame, text="油井参数", padding=5)
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        params = [
            ('杆柱长度 L (m):', 'rod_length', 800.0, 2000.0),
            ('杆柱直径 d (mm):', 'rod_diameter_mm', 16.0, 32.0),
            ('杆柱密度 ρ (kg/m³):', 'rod_density', 7000.0, 9000.0),
            ('弹性模量 E (GPa):', 'elastic_modulus_gpa', 180.0, 220.0),
            ('冲次 N (min⁻¹):', 'pump_speed', 2.0, 15.0),
            ('阻尼系数 c (s⁻¹):', 'damping_coeff', 0.01, 1.0),
        ]

        self.param_entries = {}
        for i, (label, key, vmin, vmax) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(row=i // 3, column=(i % 3) * 2, sticky='e', padx=2, pady=1)
            var = tk.DoubleVar()
            entry = ttk.Entry(param_frame, textvariable=var, width=8)
            entry.grid(row=i // 3, column=(i % 3) * 2 + 1, sticky='w', padx=2, pady=1)
            self.param_entries[key] = (var, entry)

        # 计算方法选择
        method_frame = ttk.LabelFrame(control_frame, text="计算方法", padding=5)
        method_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.method_var = tk.StringVar(value='fourier')
        ttk.Radiobutton(method_frame, text="傅里叶级数法 (Gibbs, 推荐)",
                        variable=self.method_var, value='fourier').pack(anchor='w')
        ttk.Radiobutton(method_frame, text="有限差分法 (实验性)",
                        variable=self.method_var, value='fd').pack(anchor='w')

        # 傅里叶参数
        fourier_frame = ttk.LabelFrame(control_frame, text="傅里叶参数", padding=5)
        fourier_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Label(fourier_frame, text="谐波数:").pack(side=tk.LEFT)
        self.harmonics_var = tk.IntVar(value=10)
        ttk.Spinbox(fourier_frame, from_=2, to=30, textvariable=self.harmonics_var,
                    width=5).pack(side=tk.LEFT, padx=2)

        # 计算按钮
        btn_frame = ttk.Frame(control_frame, padding=5)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Button(btn_frame, text="计算泵功图", command=self._compute).pack(
            side=tk.TOP, pady=2)
        ttk.Button(btn_frame, text="仅重构地面示功图", command=self._reconstruct_surface).pack(
            side=tk.TOP, pady=2)

        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, padx=5)

        # 图表区域 (2x2 子图)
        self.fig = Figure(figsize=(14, 8), dpi=100)

        # 地面示功图 (原始 vs 重构)
        self.ax_surface = self.fig.add_subplot(2, 3, 1)
        self.ax_surface.set_title("地面示功图 (原始 vs 傅里叶重构)")
        self.ax_surface.set_xlabel("位移 (m)")
        self.ax_surface.set_ylabel("载荷 (kN)")
        self.ax_surface.grid(True, alpha=0.3)

        # 泵功图
        self.ax_pump = self.fig.add_subplot(2, 3, 2)
        self.ax_pump.set_title("泵功图")
        self.ax_pump.set_xlabel("位移 (m)")
        self.ax_pump.set_ylabel("载荷 (kN)")
        self.ax_pump.grid(True, alpha=0.3)

        # 地面和泵功图叠加
        self.ax_compare = self.fig.add_subplot(2, 3, 3)
        self.ax_compare.set_title("地面功图 vs 泵功图")
        self.ax_compare.set_xlabel("位移 (m)")
        self.ax_compare.set_ylabel("载荷 (kN)")
        self.ax_compare.grid(True, alpha=0.3)

        # 位移-角度图
        self.ax_disp_angle = self.fig.add_subplot(2, 3, 4)
        self.ax_disp_angle.set_title("位移-曲柄转角")
        self.ax_disp_angle.set_xlabel("转角 (rad)")
        self.ax_disp_angle.set_ylabel("位移 (m)")
        self.ax_disp_angle.grid(True, alpha=0.3)

        # 载荷-角度图
        self.ax_load_angle = self.fig.add_subplot(2, 3, 5)
        self.ax_load_angle.set_title("载荷-曲柄转角")
        self.ax_load_angle.set_xlabel("转角 (rad)")
        self.ax_load_angle.set_ylabel("载荷 (kN)")
        self.ax_load_angle.grid(True, alpha=0.3)

        # 傅里叶系数幅值谱
        self.ax_spectrum = self.fig.add_subplot(2, 3, 6)
        self.ax_spectrum.set_title("傅里叶系数幅值谱")
        self.ax_spectrum.set_xlabel("谐波阶数 n")
        self.ax_spectrum.set_ylabel("幅值")
        self.ax_spectrum.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=3.0)

        # 嵌入到 Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请选择数据并点击计算")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=3)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _get_well_from_params(self):
        """从GUI参数获取油井模型"""
        p = self.param_entries
        return WellModel(
            name=self.well_var.get(),
            rod_length=p['rod_length'][0].get(),
            rod_diameter=p['rod_diameter_mm'][0].get() / 1000.0,  # mm -> m
            rod_density=p['rod_density'][0].get(),
            elastic_modulus=p['elastic_modulus_gpa'][0].get() * 1e9,  # GPa -> Pa
            pump_speed=p['pump_speed'][0].get(),
            damping_coeff=p['damping_coeff'][0].get(),
            data_points=self.well.data_points if self.well else None,
        )

    def _update_param_display(self):
        """更新参数显示"""
        if self.well is None:
            return
        p = self.param_entries
        p['rod_length'][0].set(self.well.rod_length)
        p['rod_diameter_mm'][0].set(self.well.rod_diameter * 1000)
        p['rod_density'][0].set(self.well.rod_density)
        p['elastic_modulus_gpa'][0].set(self.well.elastic_modulus / 1e9)
        p['pump_speed'][0].set(self.well.pump_speed)
        p['damping_coeff'][0].set(self.well.damping_coeff)

    def _load_default_data(self):
        """加载默认数据"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        default_file = os.path.join(base_dir, 'datasets','DK3.txt')
        if os.path.exists(default_file):
            self._load_file(default_file)
            self.well_var.set('DK3')

    def _load_file(self, filepath):
        """加载数据文件"""
        try:
            data = load_well_data(filepath)
            if len(data) < 10:
                messagebox.showerror("错误", "数据点太少")
                return

            name = os.path.splitext(os.path.basename(filepath))[0]
            params = self.DEFAULT_WELL_PARAMS.get(name, {
                'rod_length': 1000.0, 'rod_diameter': 0.022, 'pump_speed': 6.0
            })

            self.well = WellModel(
                name=name,
                rod_length=params['rod_length'],
                rod_diameter=params['rod_diameter'],
                pump_speed=params['pump_speed'],
                data_points=data,
            )
            self.current_file = filepath
            self._update_param_display()

            # 清空之前的结果
            self.surface_results = None
            self.pump_results_fourier = None
            self.pump_results_fd = None

            self.status_var.set(f"已加载: {name} ({len(data)} 个数据点)")
            self._plot_raw_data()

        except Exception as e:
            messagebox.showerror("加载失败", f"无法读取文件:\n{str(e)}")

    def _import_data(self):
        """导入数据文件"""
        filepath = filedialog.askopenfilename(
            title="选择油井数据文件",
            filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if filepath:
            self._load_file(filepath)

    def _export_results(self):
        """导出计算结果"""
        results = self.pump_results_fourier or self.pump_results_fd or self.surface_results
        if results is None:
            messagebox.showinfo("提示", "请先进行计算")
            return

        filepath = filedialog.asksaveasfilename(
            title="导出结果",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("JSON文件", "*.json")]
        )
        if not filepath:
            return

        try:
            if filepath.endswith('.json'):
                export_results_json(filepath, results)
            else:
                export_results_csv(filepath, results)
            self.status_var.set(f"结果已导出至: {filepath}")
            messagebox.showinfo("成功", "导出完成")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def _on_well_changed(self):
        """切换油井"""
        name = self.well_var.get()
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        filepath = os.path.join(base_dir,'datasets',f"{name}.txt")

        if name in self.DEFAULT_WELL_PARAMS:
            params = self.DEFAULT_WELL_PARAMS[name]
            if self.well is not None:
                self.well.name = name
                self.well.rod_length = params['rod_length']
                self.well.rod_diameter = params['rod_diameter']
                self.well.pump_speed = params['pump_speed']
                self.well.rod_area = np.pi * params['rod_diameter']**2 / 4
                self.well.wave_speed = np.sqrt(self.well.elastic_modulus / self.well.rod_density)
            self._update_param_display()

        if os.path.exists(filepath):
            self._load_file(filepath)
        else:
            messagebox.showwarning("提示", f"未找到文件: {filepath}")

    def _compute(self):
        """计算泵功图"""
        if self.well is None or self.well.data_points is None:
            messagebox.showwarning("提示", "请先加载数据")
            return

        try:
            well = self._get_well_from_params()
            n_harmonics = self.harmonics_var.get()
            method = self.method_var.get()

            self.status_var.set("正在计算...")
            self.root.update()

            # 傅里叶重构地面示功图
            self.surface_results = compute_surface_reconstruction(
                well, n_harmonics=n_harmonics)

            # 计算泵功图
            if method == 'fourier':
                self.pump_results_fourier = solve_wave_equation_fourier(
                    well, n_harmonics=n_harmonics)
                self.pump_results_fd = None
            else:
                self.pump_results_fd = solve_wave_equation_fd(well)
                self.pump_results_fourier = None

            self._plot_all_results()
            self.status_var.set(
                f"计算完成 - 方法: {'傅里叶级数法' if method == 'fourier' else '有限差分法'}, "
                f"谐波数: {n_harmonics}"
            )

        except Exception as e:
            messagebox.showerror("计算失败", f"计算过程中出错:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _reconstruct_surface(self):
        """仅重构地面示功图（不计算泵功图）"""
        if self.well is None or self.well.data_points is None:
            messagebox.showwarning("提示", "请先加载数据")
            return

        try:
            well = self._get_well_from_params()
            n_harmonics = self.harmonics_var.get()

            self.surface_results = compute_surface_reconstruction(
                well, n_harmonics=n_harmonics)
            self.pump_results_fourier = None
            self.pump_results_fd = None

            self._plot_surface_comparison()
            self.status_var.set(
                f"地面示功图重构完成 - 谐波数: {n_harmonics}")

        except Exception as e:
            messagebox.showerror("计算失败", f"计算过程中出错:\n{str(e)}")

    def _plot_raw_data(self):
        """绘制原始数据"""
        for ax in [self.ax_surface, self.ax_pump, self.ax_compare,
                    self.ax_disp_angle, self.ax_load_angle, self.ax_spectrum]:
            ax.clear()

        if self.well and self.well.data_points:
            disp = [p[0] for p in self.well.data_points]
            load = [p[1] for p in self.well.data_points]
            self.ax_surface.plot(disp, load, 'b.-', linewidth=1, markersize=3,
                                  label='原始数据')
            self.ax_surface.set_title(f"地面示功图 - {self.well.name}")
            self.ax_surface.set_xlabel("位移 (m)")
            self.ax_surface.set_ylabel("载荷 (kN)")
            self.ax_surface.legend(loc='upper right', fontsize=8)
            self.ax_surface.grid(True, alpha=0.3)

        for ax in [self.ax_pump, self.ax_compare, self.ax_disp_angle,
                    self.ax_load_angle, self.ax_spectrum]:
            ax.set_title(ax.get_title() if ax.get_title() else "")
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _plot_surface_comparison(self):
        """绘制地面示功图对比（原始 vs 傅里叶重构）"""
        for ax in [self.ax_surface, self.ax_pump, self.ax_compare,
                    self.ax_disp_angle, self.ax_load_angle, self.ax_spectrum]:
            ax.clear()

        if self.surface_results is None:
            self.canvas.draw()
            return

        r = self.surface_results
        n_harmonics = r['harmonics']

        # 地面示功图对比
        self.ax_surface.plot(r['disp_original'], r['load_original'], 'b-',
                             linewidth=1.5, label='原始数据', alpha=0.7)
        self.ax_surface.plot(r['disp_recon'], r['load_recon'], 'r--',
                             linewidth=1.5, label=f'傅里叶重构 (n={n_harmonics})')
        self.ax_surface.set_title(f"地面示功图对比 - {self.well.name}")
        self.ax_surface.set_xlabel("位移 (m)")
        self.ax_surface.set_ylabel("载荷 (kN)")
        self.ax_surface.legend(loc='upper right', fontsize=8)
        self.ax_surface.grid(True, alpha=0.3)

        # 位移-角度
        self.ax_disp_angle.plot(r['theta'], r['disp_original'], 'b-',
                                linewidth=1, label='原始', alpha=0.7)
        self.ax_disp_angle.plot(r['theta'], r['disp_recon'], 'r--',
                                linewidth=1, label='重构')
        self.ax_disp_angle.set_title("位移-曲柄转角")
        self.ax_disp_angle.set_xlabel("转角 (rad)")
        self.ax_disp_angle.set_ylabel("位移 (m)")
        self.ax_disp_angle.legend(fontsize=8)
        self.ax_disp_angle.grid(True, alpha=0.3)

        # 载荷-角度
        self.ax_load_angle.plot(r['theta'], r['load_original'], 'b-',
                                linewidth=1, label='原始', alpha=0.7)
        self.ax_load_angle.plot(r['theta'], r['load_recon'], 'r--',
                                linewidth=1, label='重构')
        self.ax_load_angle.set_title("载荷-曲柄转角")
        self.ax_load_angle.set_xlabel("转角 (rad)")
        self.ax_load_angle.set_ylabel("载荷 (kN)")
        self.ax_load_angle.legend(fontsize=8)
        self.ax_load_angle.grid(True, alpha=0.3)

        # 傅里叶频谱
        coeff = r['fourier_coeff']
        n = len(coeff['disp_an'])
        harmonics = np.arange(1, n + 1)
        disp_amp = np.sqrt(coeff['disp_an']**2 + coeff['disp_bn']**2)
        load_amp = np.sqrt(coeff['load_an']**2 + coeff['load_bn']**2)

        self.ax_spectrum.bar(harmonics - 0.15, disp_amp, width=0.3,
                             label='位移幅值', color='steelblue', alpha=0.7)
        self.ax_spectrum.bar(harmonics + 0.15, load_amp, width=0.3,
                             label='载荷幅值', color='coral', alpha=0.7)
        self.ax_spectrum.set_title("傅里叶系数幅值谱")
        self.ax_spectrum.set_xlabel("谐波阶数 n")
        self.ax_spectrum.set_ylabel("幅值")
        self.ax_spectrum.legend(fontsize=8)
        self.ax_spectrum.grid(True, alpha=0.3, axis='y')

        # 清空其他图
        for ax in [self.ax_pump, self.ax_compare]:
            ax.set_title('（请点击"计算泵功图"）')
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _plot_all_results(self):
        """绘制所有结果"""
        for ax in [self.ax_surface, self.ax_pump, self.ax_compare,
                    self.ax_disp_angle, self.ax_load_angle, self.ax_spectrum]:
            ax.clear()

        if self.surface_results is None:
            self.canvas.draw()
            return

        sr = self.surface_results
        n_harmonics = sr['harmonics']

        # 地面示功图对比
        self.ax_surface.plot(sr['disp_original'], sr['load_original'], 'b-',
                             linewidth=1.5, label='原始数据', alpha=0.7)
        self.ax_surface.plot(sr['disp_recon'], sr['load_recon'], 'r--',
                             linewidth=1.5, label=f'傅里叶重构 (n={n_harmonics})')
        self.ax_surface.set_title(f"地面示功图对比 - {self.well.name}")
        self.ax_surface.set_xlabel("位移 (m)")
        self.ax_surface.set_ylabel("载荷 (kN)")
        self.ax_surface.legend(loc='upper right', fontsize=8)
        self.ax_surface.grid(True, alpha=0.3)

        # 泵功图
        if self.pump_results_fourier is not None:
            pr = self.pump_results_fourier
            method_label = "傅里叶级数法"
        elif self.pump_results_fd is not None:
            pr = self.pump_results_fd
            method_label = "有限差分法"
        else:
            pr = None
            method_label = ""

        if pr is not None:
            self.ax_pump.plot(pr['pump_disp'], pr['pump_load'], 'g-',
                              linewidth=1.5)
            self.ax_pump.fill(pr['pump_disp'], pr['pump_load'],
                              alpha=0.2, color='green')
            self.ax_pump.set_title(f"泵功图 ({method_label})")
            self.ax_pump.set_xlabel("位移 (m)")
            self.ax_pump.set_ylabel("载荷 (kN)")
            self.ax_pump.grid(True, alpha=0.3)

            # 叠加对比图
            self.ax_compare.plot(sr['disp_original'], sr['load_original'],
                                 'b-', linewidth=1, label='地面功图', alpha=0.6)
            self.ax_compare.plot(pr['pump_disp'], pr['pump_load'],
                                 'g-', linewidth=1.5, label='泵功图')
            self.ax_compare.fill(pr['pump_disp'], pr['pump_load'],
                                 alpha=0.15, color='green')
            self.ax_compare.set_title("地面功图 vs 泵功图")
            self.ax_compare.set_xlabel("位移 (m)")
            self.ax_compare.set_ylabel("载荷 (kN)")
            self.ax_compare.legend(fontsize=8)
            self.ax_compare.grid(True, alpha=0.3)

            # 位移-角度
            self.ax_disp_angle.plot(sr['theta'], sr['disp_original'], 'b-',
                                    linewidth=1, label='地面(原始)', alpha=0.6)
            self.ax_disp_angle.plot(sr['theta'], sr['disp_recon'], 'r--',
                                    linewidth=1, label='地面(重构)', alpha=0.7)
            if 'pump_disp' in pr:
                self.ax_disp_angle.plot(pr.get('theta', sr['theta']),
                                        pr['pump_disp'], 'g-',
                                        linewidth=1.5, label='泵')
            self.ax_disp_angle.set_title("位移-曲柄转角")
            self.ax_disp_angle.set_xlabel("转角 (rad)")
            self.ax_disp_angle.set_ylabel("位移 (m)")
            self.ax_disp_angle.legend(fontsize=7)
            self.ax_disp_angle.grid(True, alpha=0.3)

            # 载荷-角度
            self.ax_load_angle.plot(sr['theta'], sr['load_original'], 'b-',
                                    linewidth=1, label='地面(原始)', alpha=0.6)
            self.ax_load_angle.plot(sr['theta'], sr['load_recon'], 'r--',
                                    linewidth=1, label='地面(重构)', alpha=0.7)
            if 'pump_load' in pr:
                self.ax_load_angle.plot(pr.get('theta', sr['theta']),
                                        pr['pump_load'], 'g-',
                                        linewidth=1.5, label='泵')
            self.ax_load_angle.set_title("载荷-曲柄转角")
            self.ax_load_angle.set_xlabel("转角 (rad)")
            self.ax_load_angle.set_ylabel("载荷 (kN)")
            self.ax_load_angle.legend(fontsize=7)
            self.ax_load_angle.grid(True, alpha=0.3)
        else:
            # 仅有地面重构的情况
            self.ax_disp_angle.plot(sr['theta'], sr['disp_original'], 'b-',
                                    linewidth=1, label='原始', alpha=0.7)
            self.ax_disp_angle.plot(sr['theta'], sr['disp_recon'], 'r--',
                                    linewidth=1, label='重构')
            self.ax_disp_angle.set_title("位移-曲柄转角")
            self.ax_disp_angle.set_xlabel("转角 (rad)")
            self.ax_disp_angle.set_ylabel("位移 (m)")
            self.ax_disp_angle.legend(fontsize=8)
            self.ax_disp_angle.grid(True, alpha=0.3)

            self.ax_load_angle.plot(sr['theta'], sr['load_original'], 'b-',
                                    linewidth=1, label='原始', alpha=0.7)
            self.ax_load_angle.plot(sr['theta'], sr['load_recon'], 'r--',
                                    linewidth=1, label='重构')
            self.ax_load_angle.set_title("载荷-曲柄转角")
            self.ax_load_angle.set_xlabel("转角 (rad)")
            self.ax_load_angle.set_ylabel("载荷 (kN)")
            self.ax_load_angle.legend(fontsize=8)
            self.ax_load_angle.grid(True, alpha=0.3)

        # 傅里叶频谱
        coeff = sr['fourier_coeff']
        n = len(coeff['disp_an'])
        harmonics = np.arange(1, n + 1)
        disp_amp = np.sqrt(coeff['disp_an']**2 + coeff['disp_bn']**2)
        load_amp = np.sqrt(coeff['load_an']**2 + coeff['load_bn']**2)

        self.ax_spectrum.bar(harmonics - 0.15, disp_amp, width=0.3,
                             label='位移幅值', color='steelblue', alpha=0.7)
        self.ax_spectrum.bar(harmonics + 0.15, load_amp, width=0.3,
                             label='载荷幅值', color='coral', alpha=0.7)
        self.ax_spectrum.set_title("傅里叶系数幅值谱")
        self.ax_spectrum.set_xlabel("谐波阶数 n")
        self.ax_spectrum.set_ylabel("幅值")
        self.ax_spectrum.legend(fontsize=8)
        self.ax_spectrum.grid(True, alpha=0.3, axis='y')

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def run(self):
        """运行应用"""
        self.root.mainloop()


# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':
    app = PumpDiagnosisApp()
    app.run()
