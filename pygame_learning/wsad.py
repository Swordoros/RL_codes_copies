import pygame
import sys

# 初始化pygame
pygame.init()

# 设置屏幕大小
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))

# 设置标题
pygame.display.set_caption("WSAD")

# 点的初始位置
point_x, point_y = screen_width // 2, screen_height // 2

# 设置颜色
white = (255, 255, 255)

# 游戏主循环
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            # 检查按键
        if event.type == pygame.KEYDOWN:
            print(f"Key pressed: {pygame.key.name(event.key)}")
            if event.key == pygame.K_w:
                point_y -= 3  # 向上移动
            elif event.key == pygame.K_s:
                point_y += 3  # 向下移动
            elif event.key == pygame.K_a:
                point_x -= 3  # 向左移动
            elif event.key == pygame.K_d:
                point_x += 3  # 向右移动
            elif event.key == pygame.K_z:
                pygame.quit()
                sys.exit()

    # 填充背景颜色
    screen.fill((0, 0, 0))

    # 绘制点
    pygame.draw.circle(screen, white, (point_x, point_y), 3)

    # 更新屏幕显示
    pygame.display.flip()

    # 控制帧率
    pygame.time.Clock().tick(60)

# 退出pygame
pygame.quit()
sys.exit()