import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Define colors
    c_input = '#2b3e50'
    c_encoder = '#4b5a6c'
    c_novel = '#b82601'
    c_head = '#2f4f4f'
    c_text = 'white'

    def draw_box(ax, x, y, width, height, text, bg_color):
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                      linewidth=1, edgecolor='black', facecolor=bg_color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                color=c_text, fontsize=12, fontweight='bold')
        return x + width/2, y, y + height

    # Inputs
    ix1, iy1_b, iy1_t = draw_box(ax, 1, 6, 2.5, 1, "Camera Images (I)", c_input)
    ix2, iy2_b, iy2_t = draw_box(ax, 8.5, 6, 2.5, 1, "LiDAR Point Cloud (P)", c_input)

    # Encoders
    ex1, ey1_b, ey1_t = draw_box(ax, 1, 4, 2.5, 1, "Camera BEV Encoder\n(LSS + Swin-T)", c_encoder)
    ex2, ey2_b, ey2_t = draw_box(ax, 8.5, 4, 2.5, 1, "LiDAR BEV Encoder\n(SparseConvNet)", c_encoder)

    # BEV Features
    fx1, fy1_b, fy1_t = draw_box(ax, 1, 2, 2.5, 1, "Camera BEV\nFeatures (F_cam)", c_input)
    fx2, fy2_b, fy2_t = draw_box(ax, 8.5, 2, 2.5, 1, "LiDAR BEV\nFeatures (F_lid)", c_input)

    # CCP
    ccp_x, ccp_yb, ccp_yt = draw_box(ax, 4.25, 4, 3.5, 1, "Cross-modal Consistency\nProbe (CCP)", c_novel)

    # Trust Maps
    tx1, ty1_b, ty1_t = draw_box(ax, 4.25, 2, 3.5, 1, "Consistency Score (S)\nModal Attribution (A)", c_input)

    # GAFM
    gafm_x, gafm_yb, gafm_yt = draw_box(ax, 4.25, 0, 3.5, 1, "Gated Adaptive\nFusion Module (GAFM)", c_novel)

    # Fused Features & Head
    ffx, ffy_b, ffy_t = draw_box(ax, 4.25, -2, 3.5, 1, "Fused BEV Features\n(F_fused)", c_input)
    dhx, dhy_b, dhy_t = draw_box(ax, 4.25, -4, 3.5, 1, "CenterPoint\nDetection Head", c_head)
    outx, outy_b, outy_t = draw_box(ax, 4.25, -6, 3.5, 1, "3D Bounding Boxes", c_input)

    # Arrows
    arrow_props = dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=2)
    
    # I -> Encoders -> Features
    ax.annotate('', xy=(ix1, ey1_t+0.1), xytext=(ix1, iy1_b-0.1), arrowprops=arrow_props)
    ax.annotate('', xy=(ix2, ey2_t+0.1), xytext=(ix2, iy2_b-0.1), arrowprops=arrow_props)
    ax.annotate('', xy=(fx1, fy1_t+0.1), xytext=(ex1, ey1_b-0.1), arrowprops=arrow_props)
    ax.annotate('', xy=(fx2, fy2_t+0.1), xytext=(ex2, ey2_b-0.1), arrowprops=arrow_props)

    # Features -> CCP
    ax.annotate('', xy=(ccp_x-1.5, ccp_yb-0.1), xytext=(fx1+1.25, fy1_t+0.5), arrowprops=arrow_props)
    ax.annotate('', xy=(ccp_x+1.5, ccp_yb-0.1), xytext=(fx2-1.25, fy2_t+0.5), arrowprops=arrow_props)

    # CCP -> Trust
    ax.annotate('', xy=(tx1, ty1_t+0.1), xytext=(ccp_x, ccp_yb-0.1), arrowprops=arrow_props)

    # Features -> GAFM
    ax.annotate('', xy=(gafm_x-1.5, gafm_yt+0.1), xytext=(fx1+1.25, fy1_b-0.1), arrowprops=arrow_props)
    ax.annotate('', xy=(gafm_x+1.5, gafm_yt+0.1), xytext=(fx2-1.25, fy2_b-0.1), arrowprops=arrow_props)
    
    # Trust -> GAFM
    ax.annotate('', xy=(gafm_x, gafm_yt+0.1), xytext=(tx1, ty1_b-0.1), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=2, linestyle='--'))

    # GAFM -> Fused -> Head -> Out
    ax.annotate('', xy=(ffx, ffy_t+0.1), xytext=(gafm_x, gafm_yb-0.1), arrowprops=arrow_props)
    ax.annotate('', xy=(dhx, dhy_t+0.1), xytext=(ffx, ffy_b-0.1), arrowprops=arrow_props)
    ax.annotate('', xy=(outx, outy_t+0.1), xytext=(dhx, dhy_b-0.1), arrowprops=arrow_props)

    ax.set_xlim(0, 12)
    ax.set_ylim(-6.5, 7.5)
    plt.tight_layout()
    plt.savefig('/Users/vinhdq1/work/CRAFT/manuscript/figures/fig_architecture.png', dpi=300, bbox_inches='tight')
    
if __name__ == '__main__':
    draw_architecture()
    print("Architecture diagram saved to manuscript/figures/fig_architecture.png")
