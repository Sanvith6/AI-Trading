import sys
sys.path.insert(0, r'C:\GRAG\AI\ai-trading')
from trading.universe.universe_manager import UniverseManager

um = UniverseManager(universe_size=50, exclusion_list=['TCS'])
eligible = um.get_eligible_instruments()
print('Eligible instruments:', len(eligible))
if eligible:
    first = eligible[0]
    print('First symbol:', first['symbol'])
    print('First score:', first['composite_score'])
    print('Universe version:', first['universe_version'])

info = um.get_universe_info()
print('Universe info:', info)

um2 = UniverseManager(universe_size=200)
eligible2 = um2.get_eligible_instruments()
print('With size 200:', len(eligible2))