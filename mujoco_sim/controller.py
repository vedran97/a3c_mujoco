class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.tracking_errors = []
        self.current_error = 0
        self.alpha = 0.01
        self.windup_guard = 500
    
    def calculate_errors(self,current_value,target,dt):
        error = target - current_value
        integral =self.integral+ error * dt
        if integral < -self.windup_guard:
            integral = -self.windup_guard
        elif integral > self.windup_guard:
            integral = self.windup_guard
        derivative = (error - self.prev_error) / dt
        return error,integral,derivative
    
    def compute(self, current_value,target, dt):
        error = target - current_value
        self.current_error = error
        self.tracking_errors.append(error)
        self.integral += error * dt
        if self.integral < -self.windup_guard:
            self.integral = -self.windup_guard
        elif self.integral > self.windup_guard:
            self.integral = self.windup_guard
        self.derivative = (error - self.prev_error) / dt
        self.control_effort = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative 
        self.prev_error = error
        return self.control_effort
    
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

    def get_state(self):
        return self.current_error,self.integral,self.derivative
    
    def get_prev_error(self):
        return self.prev_error
    
    def get_current_error(self):
        return self.current_error

    def set_alpha(self, alpha):
        self.alpha = alpha

    def reset_controller(self):
        self.integral = 0
        self.prev_error = 0
        self.tracking_errors = []
        self.current_error = 0
        self.derivative = 0
        self.control_effort = 0