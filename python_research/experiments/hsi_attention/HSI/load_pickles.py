import pickle

def main():
    try:
        sum = pickle.load(open(r"pickles\masks1_t", "rb"))
        avg = pickle.load(open(r"pickles\masks2_t", "rb"))
        return
    except Exception as e:
        print("No pickle or something else.")

if __name__ == '__main__':
    main()