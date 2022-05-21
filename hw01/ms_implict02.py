import taichi as ti

# ti.init(debug=True, arch=ti.cpu)
ti.init(arch=ti.cpu)

## 系统参数 ===============================
MAX_NUM_PARTICLES   = 256   # 最多质点数
PARTICLE_MASS       = 1     # 质点质量
BOTTOM_Y            = 0.05  # 地面位置
CONNECTION_RADIUS   = 0.15  # 质点自动连接半径
GRAVITY = ti.Vector([0, -9.8]) # 重力场
dt = 1e-3                   # 时间步长

num_particles       = ti.field(ti.i32, shape=())  # 现有质点数
spring_stiffness    = ti.field(ti.f32, shape=())  # 弹簧刚度
paused              = ti.field(ti.i32, shape=())  # 暂停
damping             = ti.field(ti.f32, shape=())  # 阻尼
# rest_length[i, j] = 0 则 i,j 未连接
rest_length = ti.field(ti.f32, shape=(MAX_NUM_PARTICLES, MAX_NUM_PARTICLES))

x = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLES) # 位置
v = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLES) # 速度

M = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(MAX_NUM_PARTICLES, MAX_NUM_PARTICLES))
J = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(MAX_NUM_PARTICLES, MAX_NUM_PARTICLES))
A = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(MAX_NUM_PARTICLES, MAX_NUM_PARTICLES))
F = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLES)
b = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLES)

r = ti.Vector.field(2, dtype=ti.f32, shape=())

spring_stiffness[None] = 10000
damping[None] = 20  # 恒定阻尼


@ti.kernel
def implicit_euler_jacobi():
    n = num_particles[None]
    k = spring_stiffness[None]
    g = GRAVITY
    I_2_2 = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
    
    # damping
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None])
        
    # update jacobi J
    # [Miles Macklin](https://blog.mmacklin.com/2012/05/04/implicitsprings/)
    # J[i, j] = ∂fi/∂xj
    # fi = ∑fik and only when k = j, ∂fik/∂xj is non-zero
    # therefore, J[i, j] = ∂fi/∂xj = ∑∂fik/∂xj = ∂fij/∂xj
    # i = j is a special case, J[i, i] = ∑∂fik/∂xk
    for i in range(n):
        J[i, i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
    for i, j in ti.ndrange(n, n):
        l_ij = rest_length[i, j]
        if (l_ij > 0):
            x_ij = x[i] - x[j]
            x_ij_norm = x_ij.norm()
            X_ij_bar = x_ij / x_ij_norm
            X_ij_mat = X_ij_bar.outer_product(X_ij_bar)
            diff = -k * ((1 - l_ij/x_ij_norm) * (I_2_2 - X_ij_mat) + X_ij_mat)
            J[i, i] += diff
            J[i, j] = -diff

    # update A
    # A = I - dt^2 * M-1 * J(t)
    for i, j in ti.ndrange(n, n):
        A[i, j] = -beta * dt2 * J[i, j] / M[i]
        if i == j:
            A[i, j] += I_2_2

    # update F
    # F = m * g + ∑ k * (x - l)
    for i in range(n):
        F[i] = M[i] * g
    for i, j in ti.ndrange(n, n):
        l_ij = rest_length[i, j]
        if l_ij > 0:
            x_ij = x[i] - x[j]
            F[i] += -k * (x_ij.norm() - l_ij) * x_ij.normalized()

    # update b
    # b = v + dt * M-1 * F    
    for i in range(n):
        b[i] = v[i] + dt * F[i] / M[i]

    # solve A * v = b
    for i in ti.static(range(5)):
        jacobi_iterate()

    # collide with the ground
    for i in range(n):
        if x[i][1] < BOTTOM_Y:
            x[i][1], v[i][1] = BOTTOM_Y, 0

    # update x
    for i in range(n):
        x[i] += v[i] * dt
        
@ti.func
def jacobi_iterate():
    n = num_particles[None]
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] @ v[j]

        # use jacobi again to solve A[i, i] * v[i] = r
        for j in ti.static(range(5)):
            v[i][0], v[i][1] = (r[0] - A[i, i][0, 1] * v[i][1]) / A[i, i][0, 0], (r[1] - A[i, i][1, 0] * v[i][0]) / A[i, i][1, 1]



def substep():
    implicit_euler_jacobi()


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32):
    "添加新质点"
    new_particle_id = num_particles[None]
    num_particles[None] += 1
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    #M[new_particle_id] = PARTICLE_MASS
    
    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < CONNECTION_RADIUS: # 与指定半径内的点建立连接
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1



new_particle(0.3, 0.3)
new_particle(0.3, 0.4)
new_particle(0.4, 0.4)


## GUI =====================================
gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)
while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1
                
    if not paused[None]:
        for step in range(10):
            substep()
    
    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)
    
    gui.line(begin=(0.0, BOTTOM_Y), end=(1.0, BOTTOM_Y), color=0x0, radius=1)
    
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.show()