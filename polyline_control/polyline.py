import torch

def advect_polylines(vector_field: torch.Tensor, polylines: torch.Tensor, dt: float) -> torch.Tensor:
    """
    vector_field: (2, ny, nx), vector_field[0] = u, vector_field[1] = v
    polylines: (num_polylines, num_points, 2)
    dt: float
    return: (num_polylines, num_points, 2), 좌표는 제한 없이 계속 확장됨
    """
    if polylines.numel() == 0:
        return polylines

    ny, nx = vector_field.shape[1], vector_field.shape[2]

    # 좌표 분리
    x = polylines[..., 0]  # (num_polylines, num_points)
    y = polylines[..., 1]

    # 속도 구할 때만 periodic index 적용
    x0 = torch.floor(x).long() % nx
    y0 = torch.floor(y).long() % ny
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny

    sx = x - torch.floor(x)
    sy = y - torch.floor(y)

    # bilinear interpolation (배치 연산)
    u = (1 - sx) * (1 - sy) * vector_field[0, y0, x0] + \
        sx * (1 - sy) * vector_field[0, y0, x1] + \
        (1 - sx) * sy * vector_field[0, y1, x0] + \
        sx * sy * vector_field[0, y1, x1]

    v = (1 - sx) * (1 - sy) * vector_field[1, y0, x0] + \
        sx * (1 - sy) * vector_field[1, y0, x1] + \
        (1 - sx) * sy * vector_field[1, y1, x0] + \
        sx * sy * vector_field[1, y1, x1]

    # 좌표는 제한하지 않고 이동
    new_x = x + u * dt
    new_y = y + v * dt

    return torch.stack([new_x, new_y], dim=-1)

def redistribute_polyline_points(polylines: torch.Tensor) -> torch.Tensor:
    """
    polylines: (num_polylines, num_points, 2)
    return: (num_polylines, num_points, 2)
    
    각 polyline의 점 수를 유지하면서 점들의 위치를 일정 간격으로 재조정.
    """
    num_polylines, num_points, _ = polylines.shape
    device = polylines.device

    # segment 길이
    diffs = polylines[:, 1:, :] - polylines[:, :-1, :]  # (B, N-1, 2)
    dists = torch.linalg.norm(diffs, dim=2)  # (B, N-1)

    # 누적 아크 길이'
    cumdist = torch.cat([torch.zeros((num_polylines,1), device=device), torch.cumsum(dists, dim=1)], dim=1)  # (B, N)

    total_len = cumdist[:, -1:]  # (B, 1)
    # 새로운 점들의 위치 (등간격)
    new_pos = torch.linspace(0, 1, num_points, device=device).unsqueeze(0) * total_len  # (B, N)

    # 각 new_pos가 속한 segment 찾기
    idx = torch.searchsorted(cumdist, new_pos, right=True) - 1
    idx = torch.clamp(idx, 0, num_points-2)

    seg_start = torch.gather(polylines, 1, idx.unsqueeze(-1).expand(-1,-1,2))
    seg_vec = torch.gather(diffs, 1, idx.unsqueeze(-1).expand(-1,-1,2))
    seg_len = torch.gather(dists, 1, idx)

    seg_offset = new_pos - torch.gather(cumdist, 1, idx)
    ratio = (seg_offset / (seg_len + 1e-8)).unsqueeze(-1)

    resampled = seg_start + seg_vec * ratio
    return resampled

def get_drawing_polylines(polylines, RESOLUTION):
    
    drawing_polylines = []
    res_tensor = torch.tensor(RESOLUTION[::-1])  # (W,H) → 텐서로 변환
    for poly in polylines:
        polyline = []
        prev_div = None  # 이전 점이 속한 cell 기록
        
        for point in poly:
            point = point.cpu()
            
            # (x,y) 좌표 → 어떤 cell인지 계산
            div = (point // res_tensor).tolist()
            
            # polyline이 끊어져야 하는 경우
            if prev_div is not None and div != prev_div:
                if polyline:  # 지금까지 모인 점 저장
                    drawing_polylines.append(polyline)
                polyline = []
            
            # 현재 cell 내부 좌표 (잔여 좌표)
            local_point = (point % res_tensor).tolist()
            polyline.append(local_point)
            
            prev_div = div
        
        if polyline:  # 마지막 잔여 선분 처리
            drawing_polylines.append(polyline)
    return drawing_polylines