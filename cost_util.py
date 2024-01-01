class APICostMonitor():
    def __init__(self, model=None):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        MODEL_COST_PER_TOKEN = {"gpt-3.5-turbo": [1e-6, 2e-6]}
        self.cost = MODEL_COST_PER_TOKEN.get(model, [1e-6, 2e-6])

    def update_usage(self, response):
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens

    def get_cost(self):
        return self.prompt_tokens*self.cost[0] + self.completion_tokens*self.cost[1]
    
    def print_cost(self):
        print(f"Cost of API Usage: {self.get_cost():.5f} USD")
