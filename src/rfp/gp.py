from webbrowser import BackgroundBrowser


class BankAccount: 

    def __init__(self, account_name: str, initial_balance: int = 0) -> None: 
        self.account_name = account_name 
        self.balance = initial_balance 

    def deposit(self, amount: int) -> None:
        self.balance += amount 
    
    def withdraw(self, amount: int) -> None: 
        self.balance -= amount 
    
    def overdrawn(self) -> bool:
        return self.balance < 0

def transfer(src: BankAccount, dst: BankAccount, amount: int) -> None:
    src.withdraw(amount)
    src.deposit(amount)

account_1 = BankAccount('Alice', 400)
account_2 = BankAccount('Bob', 200)
transfer(account_1, account_2, 50)

class AuditedBankAccount(BankAccount):
    def __init__(self, account_name: str, initial_balance: int = 0) -> None: 
        super().__init__(account_name, initial_balance)
        self.audit_log: list[str] = [] 
    
    def deposit(self, amount: int) -> None: 
        self.audit_log.append(f"Deposited {amount}")
        self.balance += amount 
    
    def withdraw(self, amount: int) -> None:
        self.audit_log.append(f"Withdrew {amount}")
        self.balance -= amount 

audited = AuditedBankAccount('Charlie', 300)
transfer(account_1, audited, 100)




# import jax 
# from jax.config import config
# config.update("jax_enable_x64", True)
# import jax.numpy as jnp 

# # def kernel(x1, x2, scale=10, z=0.25):
# #     t = -(1/2)*(x1-x2)**2*(1/z**2)
# #     return scale*jnp.exp(t)

# def kernel(x1, x2):
#     return jnp.minimum(x1, x2)

# # def kernel(x1, x2):
# #     return x1*x2

# def kernel_matrix(x):
#     t = lambda x0: jax.vmap(kernel, in_axes=(None, 0))(x0, x)
#     return jax.vmap(t)(x)

# if __name__ == '__main__':
#     import distrax 
#     import matplotlib.pyplot as plt
#     k = 10
#     x = jnp.linspace(0, 5, k)
#     mu = jnp.zeros_like(x)
#     sigma = kernel_matrix(x)
#     if k == 5:
#         print(sigma)
#     dist = distrax.MultivariateNormalFullCovariance(mu, sigma)
#     z = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(4,))
#     # print(z)
#     for i in z:
#         plt.plot(x, i)
#     plt.show()
