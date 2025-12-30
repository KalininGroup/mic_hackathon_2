def circle_point(theta, phi, center, t):
    n = np.array( [math.cos(phi_r)*math.sin(theta_r), math.sin(phi_r)*math.sin(theta_r), math.cos(phi_r)] )
    u = np.array( [-1*math.sin(phi_r), math.cos(phi_r), 0] )
    w = np.array( [math.cos(phi_r)*math.cos(theta_r), math.sin(phi_r)*math.cos(theta_r), -1*math.sin(phi_r)] )
    
    return center + r*math.cos(t)*u + r*math.sin(t)*ws