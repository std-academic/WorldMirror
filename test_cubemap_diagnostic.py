"""
Diagnostic test for cubemap face sampling correctness.

Test 1: Color-coded sectors → verify dominant color per face.
Test 2: Smooth gradient ERP → verify seam continuity (no color jumps).
Test 3: Grid-line ERP → visual inspection of distortion.
Test 4: Round-trip (perspective dirs → ERP coords → sample) consistency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image, ImageDraw
from src.utils.erp_utils import erp_to_cubemap, CUBEMAP_FACES, _sample_perspective_from_erp


def create_sector_erp(width=2048, height=1024):
    """Solid-color sectors: front=red, right=green, back=blue, left=yellow, up=cyan, down=magenta."""
    erp = np.zeros((height, width, 3), dtype=np.uint8)
    for py in range(height):
        for px in range(width):
            theta = (px / width - 0.5) * 2 * np.pi
            phi = (0.5 - py / height) * np.pi
            if phi > np.radians(45):
                erp[py, px] = [0, 255, 255]
            elif phi < np.radians(-45):
                erp[py, px] = [255, 0, 255]
            else:
                td = np.degrees(theta)
                if -45 <= td <= 45:
                    erp[py, px] = [255, 0, 0]
                elif 45 < td <= 135:
                    erp[py, px] = [0, 255, 0]
                elif -135 <= td < -45:
                    erp[py, px] = [255, 255, 0]
                else:
                    erp[py, px] = [0, 0, 255]
    return erp


def create_gradient_erp(width=2048, height=1024):
    """
    Smooth gradient ERP:
      R channel = longitude (0 at left, 255 at right → smooth wrap)
      G channel = latitude  (0 at top, 255 at bottom)
      B channel = 128 (constant)
    This makes seam discontinuities extremely obvious.
    """
    erp = np.zeros((height, width, 3), dtype=np.uint8)
    for py in range(height):
        for px in range(width):
            erp[py, px, 0] = int(px / width * 255)      # R = longitude
            erp[py, px, 1] = int(py / height * 255)      # G = latitude
            erp[py, px, 2] = 128
    return erp


def create_grid_erp(width=2048, height=1024, step_deg=15):
    """White grid lines on black background at every step_deg° of longitude/latitude."""
    erp = np.zeros((height, width, 3), dtype=np.uint8)
    for py in range(height):
        for px in range(width):
            theta_deg = (px / width - 0.5) * 360
            phi_deg = (0.5 - py / height) * 180
            on_grid = (abs(theta_deg % step_deg) < 0.8) or (abs(phi_deg % step_deg) < 0.8)
            # Color-code quadrants for orientation
            if on_grid:
                if theta_deg >= 0:
                    erp[py, px] = [255, 255, 255]  # white grid
                else:
                    erp[py, px] = [200, 200, 255]  # blue-ish grid for left half
            else:
                # Faint background color by quadrant
                if theta_deg >= 0 and phi_deg >= 0:
                    erp[py, px] = [40, 0, 0]
                elif theta_deg < 0 and phi_deg >= 0:
                    erp[py, px] = [0, 40, 0]
                elif theta_deg >= 0 and phi_deg < 0:
                    erp[py, px] = [0, 0, 40]
                else:
                    erp[py, px] = [40, 40, 0]
    return erp


def dominant_color_name(face_img):
    colors = {
        'RED (front)':     np.array([255, 0, 0]),
        'GREEN (right)':   np.array([0, 255, 0]),
        'BLUE (back)':     np.array([0, 0, 255]),
        'YELLOW (left)':   np.array([255, 255, 0]),
        'CYAN (up)':       np.array([0, 255, 255]),
        'MAGENTA (down)':  np.array([255, 0, 255]),
    }
    pixels = face_img.reshape(-1, 3).astype(float)
    best_name, best_count = None, 0
    results = {}
    for name, ref in colors.items():
        count = np.sum(np.linalg.norm(pixels - ref, axis=1) < 100)
        results[name] = count
        if count > best_count:
            best_count = count
            best_name = name
    return best_name, best_count / len(pixels), results


def stitch_cubemap_cross(faces_dict, face_size):
    """
    Stitch 6 faces into a cross layout for easy visual inspection:
    
              [up]
    [left] [front] [right] [back]
              [down]
    """
    s = face_size
    canvas = np.zeros((s * 3, s * 4, 3), dtype=np.uint8)
    canvas[:] = 50  # grey background
    
    # Row 0: up at column 1
    canvas[0:s, s:2*s] = faces_dict['up']
    # Row 1: left, front, right, back
    canvas[s:2*s, 0:s] = faces_dict['left']
    canvas[s:2*s, s:2*s] = faces_dict['front']
    canvas[s:2*s, 2*s:3*s] = faces_dict['right']
    canvas[s:2*s, 3*s:4*s] = faces_dict['back']
    # Row 2: down at column 1
    canvas[2*s:3*s, s:2*s] = faces_dict['down']
    
    return canvas


def main():
    outdir = os.path.join(os.path.dirname(__file__), "test_cubemap_output")
    os.makedirs(outdir, exist_ok=True)
    face_size = 256
    
    print("=" * 70)
    print("CUBEMAP FACE DIAGNOSTIC TEST")
    print("=" * 70)
    
    # =================== TEST 1: Sector Colors ===================
    print("\n[Test 1] Sector-colored ERP → verify dominant face colors")
    erp1 = create_sector_erp()
    Image.fromarray(erp1).save(os.path.join(outdir, "erp_sectors.png"))
    faces1 = erp_to_cubemap(erp1, face_size=face_size)
    
    expected_dominant = {
        'front': 'RED (front)', 'right': 'GREEN (right)', 'back': 'BLUE (back)',
        'left': 'YELLOW (left)', 'up': 'CYAN (up)', 'down': 'MAGENTA (down)',
    }
    test1_pass = True
    for f in faces1:
        dom, pct, _ = dominant_color_name(f['image'])
        ok = dom == expected_dominant[f['name']]
        if not ok:
            test1_pass = False
        print(f"  {'✓' if ok else '✗'} {f['name']:6s}: {dom} ({pct:.0%})")
    print(f"  → Test 1: {'PASS ✅' if test1_pass else 'FAIL ❌'}")
    
    # Save cross layout
    fd1 = {f['name']: f['image'] for f in faces1}
    cross1 = stitch_cubemap_cross(fd1, face_size)
    Image.fromarray(cross1).save(os.path.join(outdir, "cross_sectors.png"))
    
    # =================== TEST 2: Gradient Seam ===================
    print("\n[Test 2] Gradient ERP → seam continuity check")
    erp2 = create_gradient_erp()
    Image.fromarray(erp2).save(os.path.join(outdir, "erp_gradient.png"))
    faces2 = erp_to_cubemap(erp2, face_size=face_size)
    
    fd2 = {f['name']: f['image'] for f in faces2}
    for f in faces2:
        Image.fromarray(f['image']).save(os.path.join(outdir, f"grad_{f['name']}.png"))
    cross2 = stitch_cubemap_cross(fd2, face_size)
    Image.fromarray(cross2).save(os.path.join(outdir, "cross_gradient.png"))
    
    # Check horizontal seams (should be smooth gradients, no jumps)
    seams = [
        ('front', -1, 'right', 0, 'col'),   # front right col ↔ right left col
        ('right', -1, 'back',  0, 'col'),   # right right col ↔ back left col
        ('left',  -1, 'front', 0, 'col'),   # left right col ↔ front left col
        ('back',  -1, 'left',  0, 'col'),   # back right col ↔ left left col
    ]
    
    test2_pass = True
    for fa, ea, fb, eb, mode in seams:
        if mode == 'col':
            pix_a = fd2[fa][:, ea, :].astype(float)  # [H, 3]
            pix_b = fd2[fb][:, eb, :].astype(float)
        diff = np.mean(np.abs(pix_a - pix_b))
        ok = diff < 15  # allow small interpolation differences
        if not ok:
            test2_pass = False
        print(f"  {'✓' if ok else '✗'} {fa}(col {ea}) ↔ {fb}(col {eb}): avg_pixel_diff = {diff:.1f}")
    
    # Also check vertical seams (front ↔ up, front ↔ down)
    # Front top row → Up bottom row
    pix_ft = fd2['front'][0, :, :].astype(float)
    pix_ub = fd2['up'][-1, :, :].astype(float)
    diff_fu = np.mean(np.abs(pix_ft - pix_ub))
    ok = diff_fu < 15
    if not ok:
        test2_pass = False
    print(f"  {'✓' if ok else '✗'} front(row 0) ↔ up(row -1): avg_pixel_diff = {diff_fu:.1f}")
    
    pix_fb = fd2['front'][-1, :, :].astype(float)
    pix_dt = fd2['down'][0, :, :].astype(float)
    diff_fd = np.mean(np.abs(pix_fb - pix_dt))
    ok = diff_fd < 15
    if not ok:
        test2_pass = False
    print(f"  {'✓' if ok else '✗'} front(row -1) ↔ down(row 0): avg_pixel_diff = {diff_fd:.1f}")
    
    print(f"  → Test 2: {'PASS ✅' if test2_pass else 'FAIL ❌'}")
    
    # =================== TEST 3: Grid Visual ===================
    print("\n[Test 3] Grid ERP → visual inspection (saved to output)")
    erp3 = create_grid_erp()
    Image.fromarray(erp3).save(os.path.join(outdir, "erp_grid.png"))
    faces3 = erp_to_cubemap(erp3, face_size=face_size)
    fd3 = {f['name']: f['image'] for f in faces3}
    for f in faces3:
        Image.fromarray(f['image']).save(os.path.join(outdir, f"grid_{f['name']}.png"))
    cross3 = stitch_cubemap_cross(fd3, face_size)
    Image.fromarray(cross3).save(os.path.join(outdir, "cross_grid.png"))
    print("  Saved cross_grid.png — check for straight lines and no distortion")
    
    # =================== TEST 4: Round-trip direction test ===================
    print("\n[Test 4] Per-face center pixel round-trip direction test")
    test4_pass = True
    expected_z = {
        'front': [0, 0, 1], 'right': [1, 0, 0], 'back': [0, 0, -1],
        'left': [-1, 0, 0], 'up': [0, -1, 0], 'down': [0, 1, 0],
    }
    for name, R in CUBEMAP_FACES.items():
        z = R @ np.array([0, 0, 1])
        exp = np.array(expected_z[name], dtype=float)
        ok = np.allclose(z, exp, atol=1e-10)
        if not ok:
            test4_pass = False
        print(f"  {'✓' if ok else '✗'} {name:6s}: cam_Z → world {z.round(6)}, expected {exp}")
    
    # Also verify face pair orthogonality
    print("\n  Face pair orthogonality (adjacent faces should be 90° apart):")
    pairs = [('front', 'right'), ('front', 'up'), ('front', 'down'), ('right', 'back'), ('left', 'front')]
    for a, b in pairs:
        za = CUBEMAP_FACES[a] @ np.array([0, 0, 1])
        zb = CUBEMAP_FACES[b] @ np.array([0, 0, 1])
        dot = np.dot(za, zb)
        ok = abs(dot) < 1e-10
        print(f"  {'✓' if ok else '✗'} {a} · {b} = {dot:.6f} (should be 0)")
        if not ok:
            test4_pass = False
    
    print(f"  → Test 4: {'PASS ✅' if test4_pass else 'FAIL ❌'}")
    
    # =================== TEST 5: Specific pixel spot-check ===================
    print("\n[Test 5] Specific pixel spot-checks on gradient ERP")
    # In the gradient ERP: R = x/W*255, G = y/H*255, B = 128
    # For front face center pixel, it should sample from ERP center: (W/2, H/2)
    # Expected R ≈ 127, G ≈ 127, B = 128
    test5_pass = True
    c = face_size // 2
    
    checks = [
        ('front',  c, c, 127, 127, 128, "ERP center"),
        ('right',  c, c, 191, 127, 128, "ERP 75% from left"),
        ('left',   c, c,  63, 127, 128, "ERP 25% from left"),
    ]
    for fname, px, py, exp_r, exp_g, exp_b, desc in checks:
        actual = fd2[fname][py, px]
        tol = 10
        ok_r = abs(int(actual[0]) - exp_r) < tol
        ok_g = abs(int(actual[1]) - exp_g) < tol
        ok_b = abs(int(actual[2]) - exp_b) < tol
        ok = ok_r and ok_g and ok_b
        if not ok:
            test5_pass = False
        print(f"  {'✓' if ok else '✗'} {fname}({px},{py}) = {actual} expected ~({exp_r},{exp_g},{exp_b}) [{desc}]")
    
    # Back face center: ERP position at θ=π → x/W = 1.0 (wraps to 0) → R ≈ 0 or 255
    actual_back = fd2['back'][c, c]
    # θ=π maps to map_x = W, which wraps to 0: R ≈ 0
    ok = int(actual_back[0]) < 10 or int(actual_back[0]) > 245  # near 0 or 255 (ERP seam)
    if not ok:
        test5_pass = False
    print(f"  {'✓' if ok else '✗'} back({c},{c}) = {actual_back} (R should be ~0 or ~255, G ~127) [ERP seam]")
    
    # Up face center: looking at north pole → G ≈ 0
    actual_up = fd2['up'][c, c]
    ok = int(actual_up[1]) < 15
    if not ok:
        test5_pass = False
    print(f"  {'✓' if ok else '✗'} up({c},{c}) = {actual_up} (G should be ~0) [north pole]")
    
    # Down face center: looking at south pole → G ≈ 255
    actual_down = fd2['down'][c, c]
    ok = int(actual_down[1]) > 240
    if not ok:
        test5_pass = False
    print(f"  {'✓' if ok else '✗'} down({c},{c}) = {actual_down} (G should be ~255) [south pole]")
    
    print(f"  → Test 5: {'PASS ✅' if test5_pass else 'FAIL ❌'}")
    
    # =================== SUMMARY ===================
    all_pass = test1_pass and test2_pass and test4_pass and test5_pass
    print("\n" + "=" * 70)
    print(f"OVERALL: {'ALL TESTS PASSED ✅' if all_pass else 'SOME TESTS FAILED ❌'}")
    print(f"Output saved to: {outdir}/")
    print("Key files to visually inspect:")
    print("  cross_sectors.png   — color-coded sectors (should match face labels)")
    print("  cross_gradient.png  — smooth gradient (look for seam jumps)")
    print("  cross_grid.png      — grid lines (should be straight, not curved)")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
