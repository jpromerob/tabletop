import numpy as np
from math import acos,asin,atan,atan2,sin,cos


# 'r' for lengths
# 'a' for angles

'''
              //  x,y
             // \\
         r2 //   \\  r3
           //     \\
          //       \\
         //         \\
        //a2         \\a3
        \\           //
      r1 \\   r5    // r4
          \\a1-----//a4


'''


class FiveBar():
    def __init__(self,link1,link2,link3,link4,link5):
        self.r1 = link1
        self.r2 = link2
        self.r3 = link3
        self.r4 = link4
        self.r5 = link5

    def forward(self,a1,a4):
        r1 = self.r1
        r2 = self.r2
        r3 = self.r3
        r4 = self.r4
        r5 = self.r5

        A_1 = r1**2-r2**2
        B_1 = -2*r1*cos(a1)
        C_1 = -2*r4*sin(a4)

        A_2 = r5**2+r4**2-r3**2
        B_2 = -2*r5 -2*r4*cos(a1)
        C_2 = -2*r4*sin(a4)

        D = (C_1-C_2)/(B_1-B_2)
        E = (A_2-A_1)/(B_1-B_2)

        F = 2*D*E + B_1*D + C_1
        G = A_1 +B_1*E+ E**2
        D_1 = D**2

        self.y1 = (-F-np.sqrt(F**2-4*G*D_1))/(2*D_1)
        self.y2 = (-F+np.sqrt(F**2-4*G*D_1))/(2*D_1)

        self.x1 = D*self.y1 + E
        self.x2 = D*self.y2 + E


    def inverse(self,p_x,p_y):
        r1 = self.r1
        r2 = self.r2
        r3 = self.r3
        r4 = self.r4
        r5 = self.r5
        #### First Loop #####

        A1 = p_x**2 + p_y**2 + r1**2 + 2*r1*p_x - r2**2
        B1 = -4*r1*p_y
        C1 = r1**2 - 2*r1*p_x + p_x**2 + p_y**2 - r2**2

        t_11 = (-B1+np.sqrt(B1**2-4*A1*C1))/(2*A1)
        t_12 = (-B1-np.sqrt(B1**2-4*A1*C1))/(2*A1)

        self.a11 = atan2((2*t_11),(1-t_11**2))
        self.a12 = atan2((2*t_12),(1-t_12**2))

        #### Second Loop ####

        A2 = p_x**2 + p_y**2 + r5**2 + r4**2 - 2*p_x*r5 - r3**2 + 2*p_x*r4 - 2*r4*r5
        B2 = -4*p_y*r4
        C2 = p_x**2 + p_y**2 + r5**2 + r4**2 - 2*p_x*r5 - r3**2 + 2*r4*r5 - 2*p_x*r4

        t_21 = (-B2+np.sqrt(B2**2-4*A2*C2))/(2*A2)
        t_22 = (-B2-np.sqrt(B2**2-4*A2*C2))/(2*A2)

        self.a41 = atan2((2*t_21),(1-t_21**2))
        self.a42 = atan2((2*t_22),(1-t_22**2))


    def calculate_position(self, theta1, theta2):
        P1 = (0, 0)
        P2 = (self.r5, 0)
        L1 = self.r1
        L2 = self.r2
        L3 = self.r3
        L4 = self.r4
        L5 = self.r5

        # Calculate positions of the outer joints of L1 and L2
        # Assuming P1 and P2 are (x, y) tuples
        J1 = (P1[0] + L1 * np.cos(theta1), P1[1] + L1 * np.sin(theta1))
        J2 = (P2[0] + L2 * np.cos(theta2), P2[1] + L2 * np.sin(theta2))

        # Calculate distance between J1 and J2
        d = np.sqrt((J2[0] - J1[0])**2 + (J2[1] - J1[1])**2)

        if d > L3 + L4:
            raise ValueError("Linkage configuration is beyond reach.")

        # Use the law of cosines to find the angle between L3 and the line connecting J1 and J2
        # This is required to find the intersection points
        angle_L3 = np.arccos((L3**2 + d**2 - L4**2) / (2 * L3 * d))

        # Find the angle of the line connecting J1 and J2
        angle_line = np.arctan2(J2[1] - J1[1], J2[0] - J1[0])

        # Calculate the angle to reach the end-effector position
        angle_to_effector = angle_line + angle_L3

        # Calculate position of the end-effector (assuming it is at the end of L3 for simplicity)
        Ex = J1[0] + L3 * np.cos(angle_to_effector)
        Ey = J1[1] + L3 * np.sin(angle_to_effector)

        return Ex, Ey


    def get_a11(self):
        return self.a11

    def get_a12(self):
        return self.a12

    def get_a2(self,a1,p_x,p_y):
        a2 = atan2((p_y - self.r1*sin(a1)),(p_x-self.r1*cos(a1)))
        return a2

    def get_a3(self,a4,p_x,p_y):
        a3 = atan2((p_y-self.r4*sin(a4)),(p_x-self.r5-self.r4*cos(a4)))
        return a3

    def get_a41(self):
        return self.a41

    def get_a42(self):
        return self.a42

    def get_x1(self):
        return self.x1

    def get_x2(self):
        return self.x2

    def get_y1(self):
        return self.y1

    def get_y2(self):
        return self.y2