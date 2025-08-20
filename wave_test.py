# wave_builtin.py
from MutoLib import Muto
import time

bot = Muto()            # asegúrate que el puerto esté correcto en el constructor o en su config
bot.speed(2)            # 1..5 (ajusta a gusto)

print("Saludo incorporado (action 2)")
bot.action(2)           # 2 = “say hello”
time.sleep(3)           # deja que termine
bot.stop()

