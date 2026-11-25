"""plot workflow."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties, findfont
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figure/source'
BLUE = '#1F78B4'
INK = '#292929'
GREY = '#61666B'
PALE_BLUE = '#F0F6FA'
WIDTH, HEIGHT = (17.6, 9.5)
BOX_WIDTH = 3.55
TOP_Y = 7.5
BOTTOM_Y = 3.15
TEXT = {'en': {'title': 'Workflow of the proposed ER-PPO', 'titles': ['1. Household environment', '2. State observation', '3. Feasibility masking', '4. Execution classes', '5. ER-PPO policy', '6. Execution layer', '7. Reward and PPO update'], 'body': [['PV · background demand', 'EV · ESS · flexible loads', 'Price · carbon intensity'], ['Current household state $s_t$', 'Temperatures · SOC · tasks', '8-observation history $H_t$'], ['Keep legal commands', 'SOC limits · EV availability', 'Update at each interval'], ['Group by execution effects', 'Keep distinct commitments', 'Retain one representative'], ['Shared encoder · 6 device heads', 'Class scores: $\\ell_{j,r(c)}+\\log n_{j,c}$', 'Class-size correction: $q_j(c)$', 'Sample (train) / argmax (test)'], ['Execute joint commands', 'Service checks · request repair', 'Advance household state']], 'reward_title': 'Common reward $r_t$', 'reward': ['Bill · carbon · peak demand', 'Thermal comfort · service', 'Battery operation · recovery'], 'learning_title': 'Block-wise learning', 'learning': ['Collect interaction trajectories', 'Update actor and critic', 'After a full multi-day block'], 'next_state': 'Next state $s_{t+1}$', 'classes': 'Available classes', 'update': 'Update policy parameters', 'solid': 'Control loop · every 30 minutes', 'dashed': 'Learning loop · after each multi-day block'}, 'zh': {'title': '所提出的 ER-PPO 工作流程', 'titles': ['1. 家庭环境', '2. 状态观测', '3. 动态可行性掩码', '4. 执行类别构建', '5. ER-PPO 策略', '6. 执行层', '7. 奖励与 PPO 更新'], 'body': [['光伏与背景需求', '电动汽车、储能及柔性设备', '电价与碳排放强度'], ['当前家庭状态 $s_t$', '温度、SOC 与任务状态', '最近八次观测 $H_t$'], ['保留合法指令', 'SOC 边界与车辆接入状态', '每个时段动态更新'], ['按可执行效果分组', '保留不同的未来服务承诺', '每类保留一个代表'], ['共享编码器与六个设备动作头', '类别评分：$\\ell_{j,r(c)}+\\log n_{j,c}$', '类别规模修正：$q_j(c)$', '训练采样／测试取最大概率'], ['执行各设备的联合指令', '服务检查与请求修正', '更新家庭物理状态']], 'reward_title': '公共奖励 $r_t$', 'reward': ['电费、碳排与峰值需求', '热舒适与住户服务', '电池运行与库存恢复'], 'learning_title': '按区块学习', 'learning': ['收集交互轨迹', '更新策略与价值网络', '完整多日区块结束后进行'], 'next_state': '下一时刻状态 $s_{t+1}$', 'classes': '可用执行类别', 'update': '更新策略参数', 'solid': '控制闭环：每 30 分钟', 'dashed': '学习闭环：每个多日区块结束后'}}

def font(language, size, bold=False):
    family = 'Times New Roman' if language == 'en' else 'Microsoft YaHei' if bold else 'SimSun'
    prop = FontProperties(family=family, size=size, weight='bold' if bold else 'normal')
    findfont(prop, fallback_to_default=False)
    return prop

def draw(language):
    text = TEXT[language]
    fig, ax = plt.subplots(figsize=(7.25, 3.91), dpi=180)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.set_axis_off()
    fig.patch.set_facecolor('white')
    placements = [(2.3, TOP_Y, BOX_WIDTH, 2.15), (6.55, TOP_Y, BOX_WIDTH, 2.15), (10.8, TOP_Y, BOX_WIDTH, 2.15), (15.05, TOP_Y, BOX_WIDTH, 2.15), (15.05, BOTTOM_Y, BOX_WIDTH, 2.7), (10.8, BOTTOM_Y, BOX_WIDTH, 2.7), (4.425, BOTTOM_Y, 7.8, 2.7)]
    text_with_boxes = []
    patches = []

    def label(x, y, s, size=7.4, color=INK, bold=False, ha='center'):
        return ax.text(x, y, s, ha=ha, va='center', color=color, fontproperties=font(language, size, bold), zorder=6)

    def arrow(points, color=GREY, dashed=False):
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            ax.plot(xs, ys, color=color, lw=0.85, linestyle=(0, (4, 3)) if dashed else '-', solid_capstyle='butt', zorder=2)
        tail = points[-2]
        head = points[-1]
        ax.add_patch(FancyArrowPatch(tail, head, arrowstyle='-|>', mutation_scale=7.4, color=color, lw=0.85, linestyle=(0, (4, 3)) if dashed else '-', shrinkA=0, shrinkB=0, zorder=3))
    for i in range(3):
        x, y, w, h = placements[i]
        nx, ny, nw, nh = placements[i + 1]
        arrow([(x + w / 2, y), (nx - nw / 2, ny)])
    x, y, w, h = placements[3]
    px, py, pw, ph = placements[4]
    arrow([(x, y - h / 2), (px, py + ph / 2)])
    label(14.85, 5.55, text['classes'], size=6.9, color=GREY, ha='right')
    for a, b in [(4, 5), (5, 6)]:
        x, y, w, h = placements[a]
        nx, ny, nw, nh = placements[b]
        arrow([(x - w / 2, y), (nx + nw / 2, ny)])
    label(8.675, BOTTOM_Y + 0.36, '$r_t$', size=7.5)
    ex, ey, ew, eh = placements[5]
    envx, envy, envw, envh = placements[0]
    arrow([(ex, ey + eh / 2), (ex, 5.55), (envx, 5.55), (envx, envy - envh / 2)])
    label(6.55, 5.85, text['next_state'], size=7.8, color=GREY)
    update_x = 6.55
    arrow([(update_x, BOTTOM_Y - 1.35), (update_x, 0.97), (px, 0.97), (px, py - ph / 2)], color=BLUE, dashed=True)
    label(11.6, 1.25, text['update'], size=7.0, color=BLUE)
    for i, (x, y, w, h) in enumerate(placements):
        core = i in (3, 4)
        patch = Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1.0 if core else 0.8, edgecolor=BLUE if core else GREY, facecolor=PALE_BLUE if core else 'white', zorder=4)
        ax.add_patch(patch)
        patches.append(patch)
        title = label(x - w / 2 + 0.16, y + h / 2 - 0.36, text['titles'][i], size=7.8, color=BLUE if core else INK, bold=True, ha='left')
        text_with_boxes.append((title, patch))
        if i < 6:
            gap = 0.39 if i == 4 else 0.4
            start = y + h / 2 - 0.95
            for j, s in enumerate(text['body'][i]):
                artist = label(x, start - j * gap, s, size=7.0 if i == 4 else 7.25)
                text_with_boxes.append((artist, patch))
    patch = patches[-1]
    left_x, right_x = (2.48, 6.37)
    divider_x = 4.425
    ax.plot([divider_x, divider_x], [2.13, 3.89], color='#D4D9DE', lw=0.6, zorder=5)
    for cx, title, lines in [(left_x, text['reward_title'], text['reward']), (right_x, text['learning_title'], text['learning'])]:
        artist = label(cx, 3.75, title, size=7.7, bold=True)
        text_with_boxes.append((artist, patch))
        for j, s in enumerate(lines):
            artist = label(cx, 3.25 - j * 0.42, s, size=7.25)
            text_with_boxes.append((artist, patch))
    label(0.525, 9.08, text['title'], size=10.3, bold=True, ha='left')
    ax.plot([0.55, 1.18], [0.36, 0.36], color=GREY, lw=0.85)
    label(1.38, 0.36, text['solid'], size=6.9, ha='left', color=GREY)
    ax.plot([8.0, 8.63], [0.36, 0.36], color=BLUE, lw=0.85, ls=(0, (4, 3)))
    label(8.83, 0.36, text['dashed'], size=6.9, ha='left', color=BLUE)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, patch in text_with_boxes:
        box = patch.get_window_extent(renderer)
        bounds = artist.get_window_extent(renderer)
        if bounds.x0 < box.x0 + 2 or bounds.x1 > box.x1 - 2 or bounds.y0 < box.y0 + 2 or (bounds.y1 > box.y1 - 2):
            raise ValueError(f'Text exceeds its module box in {language}: {artist.get_text()}')
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f'er_ppo_workflow_{language}'
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(stem.with_suffix('.' + extension), dpi=450, facecolor='white', metadata={'Creator': 'analysis.plot_er_ppo_workflow'} if extension == 'pdf' else None)
    plt.close(fig)
    print(stem.relative_to(ROOT))

def main():
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False}):
        for language in ('en', 'zh'):
            draw(language)
    spec = {'source': 'Revision_init/修改/第三节.txt', 'control_interval_minutes': 30, 'ppo_update': 'after a complete multi-day block', 'masks': 'physical SOC bounds and EV connection; service checks are in execution', 'policy': 'shared encoder, six heads, representative logits and log(class size)', 'manuscript_inserted': False, 'captions': {'en': 'Workflow of the proposed ER-PPO-based home energy management system. Solid arrows show the half-hour control loop. Dashed arrows show policy updates from trajectories collected over a complete multi-day block.', 'zh': '所提出的基于 ER-PPO 的家庭能源管理系统工作流程。实线表示半小时控制闭环，虚线表示依据完整多日区块轨迹进行的策略更新。'}}
    (OUT / 'figure_notes.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
if __name__ == '__main__':
    main()
