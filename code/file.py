import random
import os

random.seed(42)

def main():
	x = random.random()
	pwdfile = os.getcwd()
	print("pwd of file is = {pwdfile}")
	print(f"random number = {x}")

if __name__ == '__main__':
	main()
