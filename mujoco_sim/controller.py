class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.tracking_errors = []
        
    def compute(self, current_value,target, dt):
        error = target - current_value
        self.tracking_errors.append(error)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        control_effort = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return control_effort
    
    def set_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def update_gains(self, Kp, Ki, Kd):
        self.Kp += Kp
        self.Ki += Ki
        self.Kd += Kd

    def update_gain(self, gain, amt):
        if gain == 0:
            self.Kp+=amt
        elif gain == 1:
            self.Ki+=amt
        else:
            self.Kd+=amt


    def get_error(self):
        return self.prev_error