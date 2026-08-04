for i in range(3):
    print("Ana Hasan")

a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))

sum = a + b + c

print("Sum =", sum)

str1 = input("Enter first string: ")
str2 = input("Enter second string: ")
str3 = input("Enter third string: ")

result = str1 + str2 + str3

print("Concatenated String =", result)

print("Table of 7")
for i in range(1, 11):
    print(f"7 x {i} = {7*i}")

print()

print("Table of 9")
for i in range(1, 11):
    print(f"9 x {i} = {9*i}")

n = int(input("Enter a number: "))

for i in range(1, 11):
    print(f"{n} x {i} = {n*i}")

n = int(input("Enter n: "))

sum = 0

for i in range(1, n + 1):
    sum += i

print("Sum =", sum)

a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))

if a >= b and a >= c:
    print("Maximum =", a)
elif b >= a and b >= c:
    print("Maximum =", b)
else:
    print("Maximum =", c)

n = int(input("Enter n: "))

sum = 0

for i in range(1, n + 1):
    if i % 7 == 0 and i % 9 == 0:
        sum += i

print("Sum =", sum)


n = int(input("Enter n: "))

sum = 0

for num in range(2, n + 1):
    prime = True

    for i in range(2, num // 2 + 1):
        if num % i == 0:
            prime = False
            break

    if prime:
        sum += num

print("Sum of prime numbers =", sum)


def addOdd(n):
    sum = 0

    for i in range(1, n + 1):
        if i % 2 != 0:
            sum += i

    return sum

n = int(input("Enter n: "))

print("Sum of odd numbers =", addOdd(n))


def isPrime(num):
    if num < 2:
        return False

    for i in range(2, num // 2 + 1):
        if num % i == 0:
            return False

    return True

def addPrime(n):
    sum = 0

    for i in range(2, n + 1):
        if isPrime(i):
            sum += i

    return sum

n = int(input("Enter n: "))

print("Sum of prime numbers =", addPrime(n))