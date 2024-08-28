import pygame
import sys

# 初始化pygame
pygame.init()

# 设置屏幕大小
screen_width, screen_height = 400, 300
screen = pygame.display.set_mode((screen_width, screen_height))

# 设置标题
pygame.display.set_caption("Running Spot")

# 点的初始位置
point_x, point_y = 0, screen_height // 2

# 设置颜色
color = (255, 0, 0)

j=0
vx = 4
vy = -4
ppx = []
ppy = []
# 游戏主循环
running = True
while running:
    ax = 0
    if vy>3:
        ay=-0.7
    if vy<-3:
        ay=0.7

    # 更新速度
    vy = vy + ay

    # 更新位置
    point_x = point_x + vx
    point_y = point_y + vy
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            # 检查按键
        if event.type == pygame.KEYDOWN:
            print(f"Key pressed: {pygame.key.name(event.key)}")
            if event.key == pygame.K_LEFT:
                sys.exit()
        if point_x>screen_width:
            point_x=0
        if point_x<0:
            point_x=screen_width
        if point_y>screen_height:
            point_y=0
        if point_y<0:
            point_y=screen_height


    # 填充背景颜色
    screen.fill((255, 255, 255))

    # 绘制点
    pygame.draw.circle(screen, color, (point_x, point_y), 5)
    # 留存点
    ppx.append(point_x)
    ppy.append(point_y)
    for i in range(len(ppx)):
        pygame.draw.circle(screen, (0,0,255), (ppx[i], ppy[i]), 3)
    #目标点
    pygame.draw.circle(screen, (0,0,0), (300,200), 10)
    # 更新屏幕显示
    pygame.display.flip()

    # 控制帧率
    pygame.time.Clock().tick(30)

# 退出pygame
pygame.quit()
sys.exit()