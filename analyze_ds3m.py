seq = [
    (2061000295, 'E'), (2061000296, 'E'), (2061000297, 'E'),
    (2061000298, 'O'), (2061000299, 'O'), (2061000300, 'E'),
    (2061000301, 'O'), (2061000302, 'E'), (2061000303, 'E'),
    (2061000304, 'E'), (2061000305, 'E'), (2061000306, 'O'),
    (2061000307, 'O'), (2061000308, 'O'), (2061000309, 'E'),
    (2061000310, 'E'), (2061000311, 'O'), (2061000312, 'O'),
    (2061000313, 'E'), (2061000314, 'E'), (2061000315, 'E'),
    (2061000316, 'O'), (2061000317, 'O'), (2061000318, 'E'),
    (2061000319, 'E'), (2061000320, 'O'), (2061000321, 'O'),
    (2061000322, 'O'), (2061000323, 'E'),
]

e_count = sum(1 for _, v in seq if v == 'E')
o_count = sum(1 for _, v in seq if v == 'O')
trend = 'E' if e_count >= o_count else 'O'

print(f'Total: E={e_count}  O={o_count}  => Highest repeating: {"EVEN" if trend=="E" else "ODD"}')
print(f'Always bet {trend}:')
print()
print('Round       Actual  Bet  Result  Loss_streak')
print('-' * 50)

wins = losses = 0
streak = 0
max_streak = 0

for rid, actual in seq:
    bet = trend
    if bet == actual:
        result = 'WIN'
        streak = 0
        wins += 1
    else:
        result = 'LOSS'
        streak += 1
        losses += 1
        max_streak = max(max_streak, streak)
    flag = ' <-- max' if streak == max_streak and streak > 0 else ''
    print(f'{rid}   {actual}       {bet}    {result}    {streak}{flag}')

print()
print(f'Wins: {wins}  Losses: {losses}  WR: {wins/(wins+losses)*100:.1f}%')
print(f'Max consecutive losses: {max_streak}')
