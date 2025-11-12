from typing import Any
import torch
import torch.nn.functional as F
import torch.fft as fft
import math

class Fluid:
    
    def __init__(self, shape, device='cuda', boundary_damping_width=5,
                 dx=0.01, rho_ambient=1.225, p_ambient=101325.0, T_ambient=288.15):
        """
        물리 단위를 갖는 유체 시뮬레이션
        
        Args:
            shape: 그리드 형상
            quantities: 추가 scalar 필드들
            device: 'cuda' or 'cpu'
            boundary_damping_width: 경계 감쇠 폭 (그리드 단위)
            dx: 그리드 간격 [m], 기본값 0.01m = 1cm
            rho_ambient: 주변 밀도 [kg/m³], 기본값 1.225 (해수면 공기)
            p_ambient: 주변 압력 [Pa], 기본값 101325 (1 atm)
            T_ambient: 주변 온도 [K], 기본값 288.15 (15°C)
        
        물리량 단위:
            - rho: 밀도 [kg/m³]
            - velocity: 속도 [m/s]
            - internal_E: 내부 에너지 밀도 [J/m³]
            - pressure: 압력 [Pa = N/m² = kg/(m·s²)]
        """
        #should be z(optional), y, x order.
        self.shape = shape
        self.dimensions = len(shape)
        self.device = device
        self.boundary_damping_width = boundary_damping_width
        
        # 물리 단위 파라미터
        self.dx = dx  # 그리드 간격 [m]
        self.rho_ambient = rho_ambient  # 주변 밀도 [kg/m³]
        self.p_ambient = p_ambient  # 주변 압력 [Pa]
        self.T_ambient = T_ambient  # 주변 온도 [K]
        
        # 이상기체 상수 (공기, R = R_universal / M_air)
        self.R_specific = 287.05  # [J/(kg·K)] for dry air
        
        # 음속 계산 (진단용)
        # c = sqrt(gamma * R * T) for ideal gas
        # 또는 c = sqrt(gamma * p / rho)
        
        # velocity field (dimensions, Nz, Ny, Nx)
        # dimensions is x, y, z order.
        # 단위: [m/s]
        # float32 사용: GPU 호환성 (float64는 일부 GPU에서 느리거나 불안정)
        # 큰 값 범위는 적절한 단위 스케일링으로 해결
        self.velocity = torch.zeros((self.dimensions, *shape), dtype=torch.float32, device=device)

        # compressive steps variables
        self.rho = torch.zeros(shape, dtype=torch.float32, device=device)  # [kg/m³] 밀도
        self.E = torch.zeros(shape, dtype=torch.float32, device=device)  # [J/m³] 내부 에너지 밀도
        
        # grid indices
        indices = [torch.arange(s, device=device) for s in shape]
        mesh = torch.meshgrid(*indices, indexing='ij')
        self.indices = torch.stack(mesh[::-1])
        
        # 경계 감쇠 마스크 생성 (경계로 갈수록 0에 가까워짐)
        self.boundary_mask = self._create_boundary_mask()

    def get_conserved_quantities(self):
        #보존되어야 하는 물리량들.
        total_mass = self.rho.sum().item()
        total_momentum = (self.rho * self.velocity).sum(dim=tuple(range(1, self.velocity.ndim))).detach().cpu().numpy()
        total_energy = self.E.sum().item()
        return total_mass, total_momentum, total_energy
    
    def _create_boundary_mask(self):
        """
        경계 근처에서 물리량을 감쇠시키는 마스크 생성
        중심부는 1.0, 경계로 갈수록 0.0으로 감소
        """
        mask = torch.ones(self.shape, device=self.device)
        width = self.boundary_damping_width
        
        if width <= 0:
            return mask
        
        # 각 차원에 대해 경계 감쇠 적용
        for dim_idx in range(self.dimensions):
            size = self.shape[dim_idx]
            
            # 해당 차원의 인덱스 생성
            dist_from_edge = torch.arange(size, device=self.device, dtype=torch.float32)
            
            # 각 경계로부터의 거리 계산
            dist_from_start = dist_from_edge
            dist_from_end = size - 1 - dist_from_edge
            
            # 가장 가까운 경계까지의 거리
            min_dist = torch.minimum(dist_from_start, dist_from_end)
            
            # 감쇠 함수: 경계에서 width 픽셀 내에서 smooth하게 감소
            damping = torch.clamp(min_dist / width, 0.0, 1.0)
            
            # shape에 맞게 브로드캐스트
            # damping을 올바른 차원으로 확장
            shape_for_broadcast = [1] * self.dimensions
            shape_for_broadcast[dim_idx] = size
            damping = damping.reshape(tuple(shape_for_broadcast))
            
            # 마스크에 곱하기 (모든 차원의 감쇠를 누적)
            mask = mask * damping
        
        return mask
    
    def apply_boundary_damping(self, damping_strength=0.95):
        """
        경계 근처에서 물리량을 감쇠시킴
        damping_strength: 경계에서의 감쇠 계수 (0에 가까울수록 강한 감쇠)
        """
        # 속도 감쇠
        for i in range(self.dimensions):
            self.velocity[i] = self.velocity[i] * (self.boundary_mask + (1 - self.boundary_mask) * damping_strength)
        
        # 밀도 감쇠 (너무 강하게 하면 비물리적이므로 약하게)
        self.rho = self.rho * (self.boundary_mask + (1 - self.boundary_mask) * 0.99)
        
        # 내부 에너지 감쇠
        self.internal_E = self.internal_E * (self.boundary_mask + (1 - self.boundary_mask) * 0.99)
        
    def compute_advection_grid(self, dt):
        advection_map = self.indices - self.velocity * dt
        advection_map_norm = torch.empty_like(advection_map)
        for i in range(self.dimensions):
            size = self.velocity.shape[-(i+1)]
            # 주기 경계: torch.remainder로 주기적으로 래핑
            advection_map[i] = torch.remainder(advection_map[i], size)
            advection_map_norm[i] = 2.0 * advection_map[i] / (size - 1) - 1.0
        grid = advection_map_norm.permute(list(range(1, self.dimensions + 1)) + [0]).unsqueeze(0)
        return grid
    
    def advect_field(self, field, grid, filter_epsilon=1e-2, mode='bilinear'):
        field_unsq = field.unsqueeze(0)  # (1,1,Nz,Ny,Nx)
        advected = F.grid_sample(
            field_unsq, grid, align_corners=True,
            mode=mode, padding_mode='border'
        )
        return advected.squeeze(0) * (1 - filter_epsilon) + field * filter_epsilon

    @staticmethod
    def poisson_fft(rhs, h=1.0):
        """
        Solve Laplacian p = rhs with periodic BC using FFT.
        Works for arbitrary dimensions (2D, 3D, ...).
        rhs: (...,), float32 cuda
        """
        dims = tuple(range(rhs.ndim))   # 모든 차원에서 FFT
        rhs_hat = fft.rfftn(rhs, dim=dims)

        # 각 축마다 wave numbers 생성
        ks = []
        for i, size in enumerate(rhs.shape):
            if i == rhs.ndim - 1:  # 마지막 축만 rfftfreq
                k = torch.fft.rfftfreq(size, d=h, device=rhs.device) * 2 * math.pi
            else:
                k = torch.fft.fftfreq(size, d=h, device=rhs.device) * 2 * math.pi
            ks.append(k)

        # meshgrid 만들기
        grids = torch.meshgrid(*ks, indexing="ij")

        # 일반화된 eigenvalue (라플라시안의 symbol)
        lam = torch.zeros_like(grids[0])
        for g in grids:
            lam += g**2

        lam[tuple([0] * rhs.ndim)] = 1.0  # DC 보호
        p_hat = -rhs_hat / lam
        p_hat[tuple([0] * rhs.ndim)] = 0.0

        # 역 FFT
        p = fft.irfftn(p_hat, s=rhs.shape, dim=dims)
        return p

    def get_divergence(self):
        """
        Compute divergence of vector field (주기 경계).
        velocity shape:
            2D: (2, Ny, Nx)
            3D: (3, Nz, Ny, Nx)
        
        Returns: divergence [1/s] (velocity의 공간 미분)
        """
        sum_ = torch.zeros_like(self.velocity[0])
        for i in range(self.dimensions):
            # 주기 경계: torch.roll을 사용한 중앙차분
            # ∂f/∂x = (f(x+dx) - f(x-dx)) / (2*dx)
            axis_dim = -(i + 1)
            grad = (torch.roll(self.velocity[i], -1, dims=axis_dim) - 
                   torch.roll(self.velocity[i], 1, dims=axis_dim)) / (2.0 * self.dx)
            sum_ += grad
        return sum_

    def gradient(self, field):
        """
        Compute spatial gradient of a scalar field.
        
        Args:
            field: scalar field (Ny, Nx) or (Nz, Ny, Nx)
        
        Returns: gradient vector [field_unit/m]
        """
        list_ = []
        for i in range(self.dimensions):
            # spacing=self.dx로 실제 그리드 간격 적용
            list_.append(torch.gradient(field, spacing=self.dx, dim=-(i + 1))[0])
        return torch.stack(list_)

    def pressure_projection(self):
        
        div = self.get_divergence()
        pressure = self.poisson_fft(div)
        grad_p = self.gradient(pressure)
        self.velocity -= grad_p
        return div, pressure

    def vorticity_confinement_3d(self, h=1.0, eps_conf=1.0, eps_small=1e-12):
        """
        vel: (3, Nz, Ny, Nx) tensor (float)
        returns f_conf: (3, Nz, Ny, Nx)
        """
        u = self.velocity[0]  # (Nz,Ny,Nx)
        v = self.velocity[1]
        w = self.velocity[2]
    
        # 개방 경계: torch.gradient 사용
        def d_dx(f):
            return torch.gradient(f, dim=2)[0] / h
        def d_dy(f):
            return torch.gradient(f, dim=1)[0] / h
        def d_dz(f):
            return torch.gradient(f, dim=0)[0] / h
    
        # vorticity components
        wx = d_dy(w) - d_dz(v)
        wy = d_dz(u) - d_dx(w)
        wz = d_dx(v) - d_dy(u)
    
        # magnitude (eps_small 추가하여 0 방지)
        mag = torch.sqrt(wx*wx + wy*wy + wz*wz + eps_small)
    
        # gradient of magnitude
        dmag_dx = d_dx(mag)
        dmag_dy = d_dy(mag)
        dmag_dz = d_dz(mag)
    
        # normalize
        norm_grad = torch.sqrt(dmag_dx**2 + dmag_dy**2 + dmag_dz**2 + eps_small)
        Nx = dmag_dx / norm_grad
        Ny = dmag_dy / norm_grad
        Nz = dmag_dz / norm_grad
    
        # cross product N x omega
        fx = Ny * wz - Nz * wy
        fy = Nz * wx - Nx * wz
        fz = Nx * wy - Ny * wx
    
        f_conf = eps_conf * h * torch.stack([fx, fy, fz], dim=0)
        return f_conf
    def vorticity_confinement_2d(self, eps_conf=1.0, eps_small=1e-12):
        """
        2D Vorticity Confinement Force
        velocity: (2, H, W) tensor, [vx, vy]
        eps: confinement strength
        return: (2, H, W) force field
        """
        vx, vy = self.velocity[0], self.velocity[1]
    
        # Spatial derivatives (open BC - use gradient instead of roll)
        dvx_dy = torch.gradient(vx, dim=0)[0]
        dvy_dx = torch.gradient(vy, dim=1)[0]
    
        # Vorticity (scalar in 2D)
        omega = dvy_dx - dvx_dy  
    
        # Gradient of |omega|
        abs_omega = omega.abs()
        d_abs_dy = torch.gradient(abs_omega, dim=0)[0]
        d_abs_dx = torch.gradient(abs_omega, dim=1)[0]
    
        # Normalized gradient (N)
        mag = torch.sqrt(d_abs_dx**2 + d_abs_dy**2 + eps_small)
        Nx, Ny = d_abs_dx / mag, d_abs_dy / mag
    
        # In 2D, N × ω becomes a vector in the plane:
        # f = eps * (Ny * omega, -Nx * omega)
        force_x = eps_conf * Ny * omega
        force_y = -eps_conf * Nx * omega
    
        return torch.stack([force_x, force_y], dim=0)
    def step(self, dt=1.0, nu=0.0, eps_conf=0.1):
        if(eps_conf):
            if(self.dimensions == 2):
                f_vc = self.vorticity_confinement_2d(eps_conf=eps_conf)
            else:
                f_vc = self.vorticity_confinement_3d(eps_conf=eps_conf)
            self.velocity = self.velocity + f_vc * dt
        
        
        # 1. Advect velocity
        grid = self.compute_advection_grid(dt)

        self.velocity = self.advect_field(self.velocity, grid)
        self.pressure_projection()
        
        # 경계 감쇠 적용
        self.apply_boundary_damping(damping_strength=0.95)  
    def _check_validity(self):
        # 유효성 검사: velocity, 밀도, 에너지가 유효한 값인지 확인
        # .item()을 사용해 Python boolean으로 변환해야 제대로 작동함
        has_nan_velocity = torch.isnan(self.velocity).any().item()
        has_nan_rho = torch.isnan(self.rho).any().item()
        has_nan_E = torch.isnan(self.E).any().item()
        has_inf_velocity = torch.isinf(self.velocity).any().item()
        has_inf_rho = torch.isinf(self.rho).any().item()
        has_inf_E = torch.isinf(self.E).any().item()
        has_negative_rho = (self.rho < 0).any().item()
        has_negative_E = (self.E < 0).any().item()
        if has_nan_velocity or has_nan_rho or has_nan_E:
            max_speed = torch.max(torch.norm(self.velocity, dim=0)).item() if not has_nan_velocity else float('nan')
            max_rho = self.rho.max().item() if not has_nan_rho else float('nan')
            max_E = self.E.max().item() if not has_nan_E else float('nan')
            raise ValueError(
                f"NaN detected in comp_step!\n"
                f"  - velocity NaN: {has_nan_velocity}, max_speed: {max_speed:.2f}\n"
                f"  - rho NaN: {has_nan_rho}, max_rho: {max_rho:.2f}\n"
                f"  - E NaN: {has_nan_E}, max_E: {max_E:.2f}\n"
            )
        
        if has_inf_velocity or has_inf_rho or has_inf_E:
            max_speed = torch.max(torch.norm(self.velocity, dim=0)).item() if not has_inf_velocity else float('inf')
            max_rho = self.rho.max().item() if not has_inf_rho else float('inf')
            max_E = self.E.max().item() if not has_inf_E else float('inf')
            raise ValueError(
                f"Inf detected in comp_step!\n"
                f"  - velocity Inf: {has_inf_velocity}, max_speed: {max_speed:.2f}\n"
                f"  - rho Inf: {has_inf_rho}, max_rho: {max_rho:.2f}\n"
                f"  - E Inf: {has_inf_E}, max_E: {max_E:.2f}\n"
            )
        
        if has_negative_rho or has_negative_E:
            min_rho = self.rho.min().item()
            min_E = self.E.min().item()
            raise ValueError(
                f"Negative values detected in comp_step!\n"
                f"  - rho negative: {has_negative_rho}, min_rho: {min_rho:.6f}\n"
                f"  - E negative: {has_negative_E}, min_E: {min_E:.6f}\n"
            )
    def clamp_with_tolerance(self, x, min=None, max=None, tolerance=1e-3, name=None):
        """
        torch.clamp과 유사하지만, (min 또는 max 밖으로) tolerance 이상 벗어나면 error를 raise 함.
        tolerance 이내는 clamp, tolerance 이상은 ValueError 발생
        clamp가 일어나면 print로 로그를 남김
        """
        clamped = False
        # min, max 각각 검사
        if min is not None:
            over_min = (x < min - tolerance)
            if torch.any(over_min):
                idx = torch.nonzero(over_min, as_tuple=False)
                # GPU 호환: 튜플 인덱싱 사용
                first_idx: tuple[Any, ...] = tuple(idx[0].cpu().tolist())
                val: float = x[first_idx].item()
                msg = (f"[{name}] Value {val:.6e} at {first_idx} below clamp min {min} (tolerance={tolerance})\n"
                       f"All violating indices: {idx.cpu().tolist()}")
                raise ValueError(msg)
            clamp_mask = (x < min)
            if torch.any(clamp_mask):
                clamped = True
                print(f"[clamp_with_tolerance] Clamped {torch.sum(clamp_mask).item()} elements below min {min}"
                      f" for '{name}' (tolerance={tolerance})")
            x = torch.where(clamp_mask, torch.tensor(min, dtype=x.dtype, device=x.device), x)
            
        if max is not None:
            over_max = (x > max + tolerance)
            if torch.any(over_max):
                idx = torch.nonzero(over_max, as_tuple=False)
                # GPU 호환: 튜플 인덱싱 사용
                first_idx: tuple[Any, ...] = tuple(idx[0].cpu().tolist())
                val: float = x[first_idx].item()
                msg = (f"[{name}] Value {val:.6e} at {first_idx} above clamp max {max} (tolerance={tolerance})\n"
                       f"All violating indices: {idx.cpu().tolist()}")
                raise ValueError(msg)
            clamp_mask = (x > max)
            if torch.any(clamp_mask):
                clamped = True
                print(f"[clamp_with_tolerance] Clamped {torch.sum(clamp_mask).item()} elements above max {max}"
                      f" for '{name}' (tolerance={tolerance})")
            x = torch.where(clamp_mask, torch.tensor(max, dtype=x.dtype, device=x.device), x)
            
        return x  

    def comp_step(self, dt=1e-3, gamma=1.4):
        """
        Compressible Euler step (물리 단위).
        Uses conservative form update for rho, momentum, and energy.

        Args:ㄴ
            dt: Time step [s]
            gamma: Adiabatic index (무차원), 1.4 for diatomic gas (공기)

        물리량 단위:
            rho: [kg/m³]
            u: [m/s]
            p: [Pa = N/m² = kg/(m·s²)]
            E: [J/m³ = kg/(m·s²)]
            internal_E: [J/m³]
            kinetic: [J/m³]
        """
        conserved_quantities_0 = self.get_conserved_quantities()
        rho = self.rho  # [kg/m³]
        E = self.E  # [J/m³]
        u = self.velocity  # [m/s]
        dim = self.dimensions

        kinetic = 0.5 * rho * torch.sum(u**2, dim=0)  # [J/m³]
        internal_E = E - kinetic  # [J/m³]
        p = (gamma - 1.0) * internal_E  # [Pa]

        F_rho = rho * u  # mass flux
        F_mom = rho * u.unsqueeze(1) * u.unsqueeze(0)  # momentum tensor
        for i in range(dim):
            F_mom[i, i] += p
        F_E = (E + p) * u  # energy flux
    
        # Compute spatial derivatives (divergence) with physical units
        div_F_rho = torch.zeros_like(rho)
        div_F_mom = torch.zeros_like(u)
        div_F_E = torch.zeros_like(internal_E)

        for i in range(dim):
            div_F_rho += torch.gradient(F_rho[i], spacing=self.dx, dim=-(i + 1))[0]
            for j in range(dim):
                div_F_mom[j] += torch.gradient(F_mom[j, i], spacing=self.dx, dim=-(i + 1))[0]
            div_F_E += torch.gradient(F_E[i], spacing=self.dx, dim=-(i + 1))[0]

        mom = rho * u
        rho_new = self.clamp_with_tolerance(rho - dt * div_F_rho, min=1e-6, tolerance=1e0, name="rho")
        mom_new = mom - dt * div_F_mom
        u_new = mom_new / rho_new
        E_new = self.clamp_with_tolerance(E - dt * div_F_E, min=0.0, tolerance=1e0, name="E")
        
        self.rho = rho_new
        self.velocity = u_new
        self.E = E_new
        
        self._check_validity()
 
        
        return E_new - E
      
