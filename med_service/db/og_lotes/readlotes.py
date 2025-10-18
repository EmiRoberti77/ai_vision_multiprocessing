import pandas as pd
import os

DB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class LoteDB():
    def __init__(self, path:str, sheet='Lista', code_col='Lote', usecols=None):
        self.df = pd.read_excel(path, sheet_name=sheet, engine='pyxlsb', dtype_backend='pyarrow', usecols=usecols)    
        print('set index for lote')
        if code_col in self.df.columns:
            print(f'Info:found {code_col=}')
        else:
            err = f'Err:found {code_col=}'
            print(err)
            raise Exception(err)

        self.df.set_index(code_col, inplace=True)
        print(self.df.head())

    def find_lote(self, lote:str):
        try:
            result = self.df.loc[lote]
            print(f'Lote result={result}') 
            return result
        except KeyError:
            print(f"Not found:{lote} in lote_db")
            return None
        except Exception as e:
            print(f"Err:{str(e)} [{lote}]")
            return None
         

# lote_db = LoteDB(path=os.path.join(DB_ROOT, 'Listado de lotes.xlsb'))
# result = lote_db.find_lote('221563N')
# print(result['SKU'])