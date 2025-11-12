import torch
import torch.nn.functional as F
import torch.fft as fft
import math

class Fluid:
    def __init__(self, shape, *quantities, device='cuda'):
        #should be z(optional), y, x order.
        self.shape = shape
        self.dimensions = len(shape)
        self.device = device

        # velocity field (dimensions, Nz, Ny, Nx)
        # dimensions is x, y, z order.
        self.velocity = torch.zeros((self.dimensions, *shape), dtype=torch.float32, device=device)
        
        self.quantities = quantities
        for q in quantities:
            setattr(self, q, torch.zeros(shape, dtype=torch.float32, device=device))

        # compressive steps variables
        self.rho = torch.zeros(shape, dtype=torch.float32, device=device)
        self.E = torch.zeros(shape, dtype=torch.float32, device=device)
        
        # grid indices
        indices = [torch.arange(s, device=device) for s in shape]
        mesh = torch.meshgrid(*indices, indexing='ij')
        self.indices = torch.stack(mesh[::-1])
        
    def compute_advection_grid(self, dt):
        advection_map = self.indices - self.velocity * dt
        advection_map_norm = torch.empty_like(advection_map)
        for i in range(self.dimensions):
            size = self.velocity.shape[-(i+1)]
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
        Compute divergence of vector field.
        velocity shape:
            2D: (2, Ny, Nx)
            3D: (3, Nz, Ny, Nx)
        """
        sum_ = torch.zeros_like(self.velocity[0])
        for i in range(self.dimensions):
            grad = torch.gradient(self.velocity[i], dim= -(i + 1))[0]
            sum_ += grad
        return sum_

    def gradient(self, field):
        list_ = []
        for i in range(self.dimensions):
            list_.append(torch.gradient(field, dim=-(i + 1))[0])
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
    
        # 중앙차분: 축 0=z, 1=y, 2=x
        def d_dx(f):
            return (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2.0 * h)
        def d_dy(f):
            return (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2.0 * h)
        def d_dz(f):
            return (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2.0 * h)
    
        # vorticity components
        wx = d_dy(w) - d_dz(v)
        wy = d_dz(u) - d_dx(w)
        wz = d_dx(v) - d_dy(u)
    
        # magnitude
        mag = torch.sqrt(wx*wx + wy*wy + wz*wz)
    
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
    
        # Spatial derivatives (periodic BC with torch.roll)
        dvx_dy = (torch.roll(vx, -1, dims=0) - torch.roll(vx, 1, dims=0)) * 0.5
        dvy_dx = (torch.roll(vy, -1, dims=1) - torch.roll(vy, 1, dims=1)) * 0.5
    
        # Vorticity (scalar in 2D)
        omega = dvy_dx - dvx_dy  
    
        # Gradient of |omega|
        abs_omega = omega.abs()
        d_abs_dy = (torch.roll(abs_omega, -1, dims=0) - torch.roll(abs_omega, 1, dims=0)) * 0.5
        d_abs_dx = (torch.roll(abs_omega, -1, dims=1) - torch.roll(abs_omega, 1, dims=1)) * 0.5
    
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
        for q in self.quantities:
            setattr(self, q, self.advect_field(getattr(self, q).unsqueeze(0), grid).squeeze(0))

        self.pressure_projection()  

    def comp_step(self, dt=1e-3, gamma=1.4):
        """
        Compressible Euler step.
        Uses conservative form update for rho, momentum, and energy.
        """

        rho = self.rho
        E = self.E
        u = self.velocity
        dim = self.dimensions
    
        # Pressure from state equation
        kinetic = 0.5 * torch.sum(u**2, dim=0)
        p = (gamma - 1.0) * (E - rho * kinetic)
    
        # Fluxes
        F_rho = rho * u                          # mass flux
        F_mom = rho * u.unsqueeze(1) * u.unsqueeze(0)  # momentum tensor
        for i in range(dim):
            F_mom[i, i] += p                     # add pressure on diagonal
        F_E = (E + p) * u                        # energy flux
    
        # Compute spatial derivatives (simple central diff)
        def grad(f, axis):
            return (torch.roll(f, -1, dims=-(axis+1)) - torch.roll(f, 1, dims=-(axis+1))) * 0.5
    
        div_F_rho = torch.zeros_like(rho)
        div_F_mom = torch.zeros_like(u)
        div_F_E = torch.zeros_like(E)
    
        for i in range(dim):
            div_F_rho += grad(F_rho[i], i)
            for j in range(dim):
                div_F_mom[j] += grad(F_mom[j, i], i)
            div_F_E += grad(F_E[i], i)
    
        # Explicit Euler update
        rho = rho - dt * div_F_rho
        mom = rho * u - dt * div_F_mom
        E = E - dt * div_F_E
    
        # Update variables
        self.velocity = mom / (rho + 1e-10)
        self.rho = rho
        self.E = E
        
        grid = self.compute_advection_grid(dt)

        self.velocity = self.advect_field(self.velocity, grid)
        for q in self.quantities:
            setattr(self, q, self.advect_field(getattr(self, q).unsqueeze(0), grid).squeeze(0))
    
        return p

        