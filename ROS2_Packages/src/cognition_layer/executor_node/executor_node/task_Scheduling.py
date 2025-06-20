import math
from typing import List, Dict, Tuple, Optional

class Task:
    def __init__(self, task_id: int, value: float, exec_time: float,
                 deadline: float, period: float, release_time: float):
        self.task_id = task_id
        self.value = value              # 任务预期价值
        self.exec_time = exec_time      # 理论执行时间
        self.deadline = deadline        # 相对截止期
        self.period = period            # 执行周期
        self.release_time = release_time    # 放行时间
        self.abs_deadline = release_time + deadline  # 绝对截止期

        self.remaining_exec_time = exec_time    # 剩余执行时间
        self.waiting_time = 0                   # 等待时间
        self.executed_time = 0                  # 已执行时间
        self.state = "sleep"

    def update_state(self, current_time: float) -> None:
        if self.state == "sleep" and current_time >= self.release_time:
            self.state = "waiting"
    
    def calculate_remaining_value_density(self, p: float) -> float:
        """计算剩余价值密度 RVD"""
        if self.remaining_exec_time<=0:
            return 0.0
        
        # IVi(t) = (Vi * t^p) / Ci^p
        immediate_value = (self.value * (self.executed_time**p))/(self.exec_time**p)
        remaining_value = self.value - immediate_value

        return remaining_value / self.remaining_exec_time

    def calculate_execution_urgency(self, current_time: float, q: float) -> float:
        """计算执行紧迫性 δ"""
        if self.state == "executing" or self.state == "active":
            if hasattr(self, "urgency_at_start"):
                return self.urgency_at_start
            else:
                slack_time = self.abs_deadline - current_time
                if slack_time <= 0:
                    return q
                execution_intensity = self.remaining_exec_time / slack_time
                self.urgency_at_start = q ** execution_intensity
                return self.urgency_at_start
        
        slack_time = self.abs_deadline - current_time
        if slack_time <= 0:
            return q
        
        execution_intensity = self.remaining_exec_time / slack_time
        execution_intensity = min(execution_intensity, 1.0)
        return q**execution_intensity
    
    def calculate_dynamic_priority(self, current_time: float,  p: float, q: float) -> float:
        # DyPri = RVD * δ 动态优先级
        if self.state == "sleep":
            return 0.0
        
        rvd = self.calculate_remaining_value_density(p)
        urgency = self.calculate_execution_urgency(current_time, q)
        return rvd * urgency
    
class DRTPScheduler:
    def __init__(self, p: float=2.0, q: float=2.0, beta: float=3.0):
        self.p = p
        self.q = q
        self.beta = beta
        self.current_time = 0.0
        self.executing_task = None
        self.task_queue = []
        self.completed_tasks = []
        self.aborted_tasks = []

    def add_task(self, task: Task) -> None:
        self.task_queue.append(task)
    
    def update_task_states(self) -> None:
        self.task_queue = [task for task in self.task_queue 
                       if task.state not in ["completed", "aborted"]]
    
        for task in self.task_queue:
            task.update_state(self.current_time)
        
        if self.executing_task:
            self.executing_task.state = "executing"


    def find_highest_priority_task(self) -> Optional[Task]:
        highest_priority = -1
        highest_priority_task = None

        for task in self.task_queue:
            if task.state in ["waiting", "active"]:
                priority = task.calculate_dynamic_priority(self.current_time, self.p, self.q)
                if priority > highest_priority:
                    highest_priority = priority
                    highest_priority_task = task
        return highest_priority_task
    
    def check_preemption_condition(self, task: Task) -> bool:
        if not self.executing_task:
            return True
        
        exec_priority = self.executing_task.calculate_dynamic_priority(self.current_time, self.p, self.q)
        task_priority = task.calculate_dynamic_priority(self.current_time, self.p, self.q)
        
        return task_priority >= self.beta * exec_priority


    def check_deadline_conditions(self, t1: Task, t2: Task) -> Dict[str, bool]:
        results = {
            "t2_can_wait" : False,
            "t1_can_wait" : False,
            "t1_can_complete_after_t2": False
        }

        t1_remaining = t1.remaining_exec_time
        t2_remaining = t2.remaining_exec_time
        
        # d2 - τi - (C1 - t1) ≥ C2 - t2
        results["t2_can_wait"] = (t2.abs_deadline - self.current_time - t1_remaining) >= t2_remaining
        # d1 - τi - (C2 - t2) ≥ C1 - t1
        results["t1_can_wait"] = (t1.abs_deadline - self.current_time - t2_remaining) >= t1_remaining
        
        # 条件(b)(i): t1被抢占后等待t2完成仍能满足截止期
        results["t1_can_complete_after_t2"] = (t1.abs_deadline - self.current_time - t2_remaining) >= t1_remaining
        
        return results

    def strategy_i(self, t1: Task, t2: Task) -> None:
        t1_remaining = t1.remaining_exec_time
        t2_remaining = t2.remaining_exec_time


        t1_priority_after = t1.calculate_dynamic_priority(self.current_time + t2_remaining, self.p, self.q)
        t2_priority_after = t2.calculate_dynamic_priority(self.current_time + t1_remaining, self.p, self.q)
        
        self.execute_task(t1, t1_remaining)
        
        t2.waiting_time += t1_remaining
        t2_priority = max(t1_priority_after, t2_priority_after)
        t2.__dict__["dynamic_priority"] = t2_priority
    
    def strategy_ii(self, t1: Task, t2: Task) -> None:
        t1.state = "active"
        t1.waiting_time += self.current_time - t1.release_time
        self.task_queue.append(t1)
        
        self.execute_task(t2, t2.remaining_exec_time)
        
        t1.release_time = self.current_time
        t1.waiting_time = 0
    
    def strategy_iii(self, t1: Task, t2: Task) -> bool:
        t1_immediate_value = (t1.value * (t1.executed_time ** self.p)) / (t1.exec_time ** self.p)
        
        t2_rvd = t2.calculate_remaining_value_density(self.p)
        t2_urgency = t2.calculate_execution_urgency(self.current_time, self.q)
        t2_compensated_value = t2.value - t1_immediate_value
        
        t1_rvd = t1.calculate_remaining_value_density(self.p)
        t1_urgency = t1.calculate_execution_urgency(self.current_time, self.q)
        
        # 补偿条件判断
        left_side = t2_rvd * t2_compensated_value * t2_urgency
        right_side = self.beta * t1_rvd * t1.value * t1_urgency
        
        if left_side >= right_side:
            print(f"策略III：任务{t2.task_id}抢占并夭折任务{t1.task_id}")

            t1.state = "aborted"
            self.aborted_tasks.append(t1)
            
            self.execute_task(t2, t2.remaining_exec_time)
            return True
        else:
            print(f"策略III：任务{t2.task_id}不满足补偿条件，不抢占")
            return False
    
    def execute_task(self, task: Task, execution_time: float) -> None:
        """执行任务指定时间"""
        print(f"时间{self.current_time:.2f}：任务{task.task_id}开始执行，剩余执行时间{execution_time:.2f}")
        
        self.current_time += execution_time
        task.executed_time += execution_time
        task.remaining_exec_time -= execution_time
        
        if task.remaining_exec_time <= 0:
            task.state = "completed"
            self.completed_tasks.append(task)
            print(f"时间{self.current_time:.2f}：任务{task.task_id}执行完成")
            
            if self.executing_task == task:
                self.executing_task = None
        else:
            task.state = "executing"
            print(f"时间{self.current_time:.2f}：任务{task.task_id}暂停执行，剩余执行时间{task.remaining_exec_time:.2f}")
    
    def schedule(self, total_time: float) -> None:
        while self.current_time < total_time:
            self.update_task_states()
            
            highest_priority_task = self.find_highest_priority_task()
            
            if not self.executing_task and highest_priority_task:
                self.executing_task = highest_priority_task
                self.task_queue.remove(highest_priority_task)
                self.execute_task(self.executing_task, min(self.executing_task.remaining_exec_time, 
                                                         total_time - self.current_time))
                continue
            
            if self.executing_task and highest_priority_task:
                if self.check_preemption_condition(highest_priority_task):
                    conditions = self.check_deadline_conditions(self.executing_task, highest_priority_task)
                    
                    if conditions["t2_can_wait"] and conditions["t1_can_wait"]:
                        self.strategy_i(self.executing_task, highest_priority_task)
                    else:
                        if conditions["t1_can_complete_after_t2"]:
                            self.strategy_ii(self.executing_task, highest_priority_task)
                        else:
                            self.strategy_iii(self.executing_task, highest_priority_task)
                    
                    self.executing_task = None
                    highest_priority_task = self.find_highest_priority_task()
                    if highest_priority_task:
                        self.executing_task = highest_priority_task
                        self.task_queue.remove(highest_priority_task)
                else:
                    self.execute_task(self.executing_task, min(self.executing_task.remaining_exec_time, 
                                                             total_time - self.current_time))
            elif self.executing_task:
                self.execute_task(self.executing_task, min(self.executing_task.remaining_exec_time, 
                                                         total_time - self.current_time))
            else:
                self.current_time += 1.0
        
        self.print_scheduling_results()
    
    def print_scheduling_results(self) -> None:
        total_value = sum(task.value for task in self.completed_tasks)
        miss_ratio = len(self.aborted_tasks) / (len(self.completed_tasks) + len(self.aborted_tasks)) if \
                     (len(self.completed_tasks) + len(self.aborted_tasks)) > 0 else 0
        
        print("\n===== DRTP调度算法执行结果 =====")
        print(f"总执行时间: {self.current_time:.2f}")
        print(f"完成任务数: {len(self.completed_tasks)}")
        print(f"夭折任务数: {len(self.aborted_tasks)}")
        print(f"系统累积价值收益: {total_value:.2f}")
        print(f"任务截止期错失率: {miss_ratio:.4f}")

# def example_drtp_scheduling():
#     """DRTP调度算法示例"""
#     tasks = [
#         Task(1, 60, 9, 11, 16, 0),
#         Task(2, 80, 7, 8, 11, 0),
#         Task(3, 40, 2, 7, 8, 0),
#         Task(4, 30, 5, 6, 9, 0)
#     ]
    
#     scheduler = DRTPScheduler(p=2.0, q=2.0, beta=3.0)
#     for task in tasks:
#         scheduler.add_task(task)
    
#     scheduler.schedule(total_time=50.0)

def create_function_task(func, task_id, value, exec_time, deadline, period):
    """创建函数任务"""
    def execute_function():
        print(f"开始执行函数 {func.__name__}")
        func()
        print(f"函数 {func.__name__} 执行完成")
    
    # 创建一个特殊的任务，其执行会调用对应的函数
    class FunctionTask(Task):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.execute_function = execute_function
    
    return FunctionTask(task_id, value, exec_time, deadline, period, release_time=0)
