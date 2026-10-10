#1, i
from numpy import double, long


roll = input("Enter your 10-digit roll number: ")

L = []

for digit in roll:
    L.append(int(digit) * 10)

print("List L =", L)

#1, ii
n = int(input("Enter insert no: "))
m = int(input("Enter insert no2: "))
L.append(n)
L.insert(1, m)
print("List L after insertion =", L)
print("List L after insertion 2 =", L)

#1, iii
L.remove(n)
L.pop(1)
print("List L after removal and popping =", L)

#1, iv
L.sort()
print("List L after sorting =", L)
L.sort(reverse=True)
print("List L after sorting in reverse =", L)

#1, v
