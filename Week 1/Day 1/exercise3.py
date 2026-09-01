# Exercise 3: Temperature Converter

celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32

print(f"\n--- Temperature Conversion ---")
print(f"{celsius}°C = {fahrenheit}°F")

# Additional conversions for learning
kelvin = celsius + 273.15
print(f"{celsius}°C = {kelvin}K")
