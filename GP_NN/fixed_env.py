import numpy as np
import math
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import Box2D


from gymnasium.envs.registration import register

FIXED_HEIGHT = 2.0



# 引用原始环境中的常量
from gymnasium.envs.box2d.lunar_lander import (
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
    LEG_AWAY,
    LEG_DOWN,
    LEG_W,
    LEG_H,
    LEG_SPRING_TORQUE,
    LANDER_POLY
)

class ContactDetector(Box2D.b2ContactListener):
    def __init__(self, env):
        Box2D.b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]

        # 检查腿部接触
        for i in range(2):
            if self.env.legs[i] in bodies:
                self.env.legs[i].ground_contact = True

        # 检查 lander 本体直接撞地（表示坠毁）
            if self.env.lander in bodies and not any(self.env.legs[i] in bodies for i in range(2)):
                self.env.game_over = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class FixedLander(LunarLander):
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()

        from Box2D import b2World, b2FixtureDef as fixtureDef, b2PolygonShape as polygonShape, b2EdgeShape as edgeShape, b2RevoluteJointDef as revoluteJointDef

        self.world = b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # 地形（去除地形高度随机性）
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        for i in [-2, -1, 0, 1, 2]:
            height[CHUNKS // 2 + i] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i] + height[i + 1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # 着陆器主体
        initial_x = VIEWPORT_W / SCALE / 2
        #initial_y = VIEWPORT_H / SCALE
        initial_y = self.helipad_y + FIXED_HEIGHT
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # 着陆腿
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)

            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # 左右方向不同
            )
            if i == -1:
                rjd.lowerAngle = +0.4
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.4
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()

        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}




register(
    id="FixedLander-v3",
    entry_point="fixed_env:FixedLander",  # 模块名:类名
    max_episode_steps=400,
)