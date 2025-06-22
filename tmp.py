#!/usr/bin/env python3
def main():
    while True:
        line = input()
        if line == 'uci':
            print("id name MyPythonBot")
            print("id author You")
            print("uciok")
        elif line == 'isready':
            print("readyok")
        elif line.startswith('position'):
            # parse position
            pass
        elif line.startswith('go'):
            # respond with bestmove
            print("bestmove e2e4")
        elif line == 'quit':
            break

if __name__ == "__main__":
    main()