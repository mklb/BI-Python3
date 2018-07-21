
class CostBenefitMatrix:
    def __init__(self, good_customer, loss):
        self.cost_benefit_info = [[0 for x in range(2)] for y in range(2)]
        self.cost_benefit_info[0][0] = good_customer-0          # wir sagen er überlebt und er überlebt wirklich -> versicherung hat wenig geld bekommen + muss nichts zahlen
        self.cost_benefit_info[0][1] = good_customer-loss     # wir sagen er überlebt und er überlebt nicht    -> versicherung hat wenig geld bekommen + muss viel zahlen
        self.cost_benefit_info[1][0] = 0-good_customer          # wir sagen er überlebt nicht und er überlebt    -> versicherung hat viel geld bekommen + muss nichts zahlen
        self.cost_benefit_info[1][1] = 0     # wir sagen er überlebt nicht und er stirbt      -> versicherung hat viel geld bekommen + muss viel zahlen
    
    def get_cost_benefit_matrix_info(self):
        return self.cost_benefit_info
