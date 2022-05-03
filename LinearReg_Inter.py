from web3 import Web3, HTTPProvider
import web3
import json
import datetime

class RopEth():
    def __init__(self):
        API_URL = "https://eth-ropsten.alchemyapi.io/v2/WIYEPYWikXeubjVSB1lnQB9RaZBa5oNP"
        PRIVATE_KEY = "7865c3bdd68113e704cea5a8fb002fda782005ab1a4680a7fbae31f01d320eb3"
        PUBLIC_KEY = "0x4f165218486CAE53022802701882b52d108076C3"
        self.contract_address = "0xe7220113Eba4449eEB7e2979562711b6dD121703"  #linear regression sc
        self.wallet_private_key = PRIVATE_KEY
        self.wallet_address = PUBLIC_KEY
        self.w3 = Web3(HTTPProvider(API_URL))
        with open('contracts/LinearReg.json') as f:
            self.data = json.load(f)
            # print("Contract ABI: ",self.data["abi"])
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.data["abi"])

    def model_coeff_get(self):
        # Executing a transaction.
        nonce = self.w3.eth.get_transaction_count(self.wallet_address)
    
        estimatedGas = self.contract.functions.model_show().estimateGas()
        # print("Estimated gas to execute the transaction: ",estimatedGas)
        # print(dir(self.contract.functions.update(message)))
        txn_dict = self.contract.functions.model_show().buildTransaction({
            'gas': estimatedGas,
            'from': self.wallet_address,
            'nonce': nonce,
        })
        print(txn_dict)
        # print(dir(self.w3.eth.account))
        signPromise = self.w3.eth.account.signTransaction(txn_dict, self.wallet_private_key)
        # print(dir(signPromise))
        result = self.w3.eth.send_raw_transaction(signPromise.rawTransaction)
        return result

    def result_get(self):
        v = self.contract.functions.model_show().call()
        w = self.contract.functions.simple_linear_regression().call() 
        #print(dir(self.contract.functions()))
        print("The coefficients are: ", v)
        print("The prediction for the first 5 items: ", w)
        
    


if __name__ == '__main__':
    r =RopEth()
    #r.model_coeff_get()
    r.result_get()
    #print("Current data from contract: ", r.getLatestData())