# coding=utf-8
"""
MIT License

Copyright (c) 2017 Cyrille Rossant

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt

w = 400
h = 300

"""
Configuación de luces, cuantas más luces pongamos, más clara será la escena y será
más difícil diferenciar los distintos colores de las luces
"""
# Light position and color.
L1 = np.array([7., 3., -10.])
color_light1 = np.array([0., 1., 0.]) # green light
L2 = np.array([-15., 7., -10.])
color_light2 = np.array([1., 0., 0.])  # red light
L3 = np.array([1., 10., -3.])
color_light3 = np.array([0., 0., 1.])  # blue light

lights = np.array([L1,L2,L3])
colors = np.array([color_light1, color_light2, color_light3])

#Número vertices para el triangle strip, para dibujar un triangulo num_vert debe ser >= 3
num_vert = 9


def normalize(x):
    x /= np.linalg.norm(x)
    return x


def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d


def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def intersect_triangle(O, D, P, N):

    p_intsc = intersect_plane(0,D,P[0],N);
    if (p_intsc == np.inf):
        return p_intsc

    #Vértices del triángulo
    A = P[0]
    B = P[1]
    C = P[2]

    wO = O - A
    a = -np.dot(N,wO)
    b = np.dot(N,D)
    r = a / b
    if (r < 1e-6):
        return np.inf

    I = O + r*D

    #Vectores del tríangulo
    u = B - A
    v = C - A
    w = I - A

    #Cálculo del producto de puntos
    dot_uu = np.dot(u,u)
    dot_uv = np.dot(u,v)
    dot_vv = np.dot(v,v)
    dot_wu = np.dot(w,u)
    dot_wv = np.dot(w,v)

    D = dot_uv * dot_uv - dot_uu * dot_vv
    U = (dot_uv * dot_wv - dot_vv * dot_wu) / D
    V = (dot_uv * dot_wu - dot_uu * dot_wv) / D

    if (U >= 0.) and (V >= 0.) and (U + V < 1.):
        return 1.
    else:
        return np.inf


def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj['position'], obj['normal'])


def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        N = obj['normal']
    return N


def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color


def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    # Start computing the color.
    col_ray = ambient
    i = 0
    for i in range(len(lights)):
        toL = normalize(lights[i] - M)
        toO = normalize(O - M)
        # Shadow: find if the point is shadowed or not.
        l = [intersect(M + N * .0001, toL, obj_sh)
            for k, obj_sh in enumerate(scene) if k != obj_idx]
        if l and min(l) < np.inf:
            return
        else:
            # Lambert shading (diffuse).
            col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
            # Blinn-Phong shading (specular).
            col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * colors[i]
    return obj, M, N, col_ray


def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
                radius=np.array(radius), color=np.array(color), reflection=.5)


def add_plane(position, normal):
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=lambda M: (color_plane0
                                 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
                diffuse_c=.75, specular_c=.5, reflection=.25)


def add_triangle(position, color):

    u = np.subtract(position[1], position[0])
    v = np.subtract(position[2], position[1])
    normal = (np.cross(u, v))

    return dict(type='triangle', position=position,
        color=np.array(color), reflection=.5, normal=normal)


def add_triangle_strip (vertex, color):

    if vertex < 3: return []

    starting = np.array([[-1.25,.75,0.5],[-1.,1.,0.5],[-.75,.75,0.5]])

    strip = [add_triangle(starting, color)]
    distance = starting[1, 0] - starting[0, 0]

    for i in range(3, vertex):
        i = i + 1
        x = starting[2,0] + distance
        y = 0.75
        if i % 2 == 0 : y = 1.
        z = 0.5
        vertex_pos = [x,y,z]

        if i % 2 == 0:
            triangle = add_triangle([starting[1], vertex_pos, starting[2]], [0.,0.,1.])
        else:
            triangle = add_triangle([starting[1], starting[2], vertex_pos], color)

        starting = np.array([starting[1], starting[2], vertex_pos])
        strip.append(triangle)

    return strip


# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)

triangle1_pos = np.array([[-1.,-0.5,0.5],[-0.5,0.5,0.5],[0.,-0.5,0.5]])

strip = add_triangle_strip(num_vert,[0.,1.,0.])

scene = [add_triangle(triangle1_pos,[1.,0.,0.]),
         add_sphere([.75, .1, 1.], .6, [0., 0., 0.]),
         add_plane([0., -.5, 0.], [0., 1., 0.]),
         ]
scene = strip + scene

# Default light and material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 5  # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
O = np.array([0., 0.35, -1.])  # Camera.
Q = np.array([0., 0., 0.])  # Camera pointing to.
img = np.zeros((h, w, 3))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print i / float(w) * 100, "%"
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)
        depth = 0
        rayO, rayD = O, D
        reflection = 1.
        # Loop through initial and secondary rays.
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            # Reflection: create a new ray.
            rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
            depth += 1
            col += reflection * col_ray
            reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('fig.png', img)
