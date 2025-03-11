import os
import arguably

@arguably.command
def run(config: str = "default.yaml"):
    print(config)

if __name__ == "__main__":
    arguably.run()