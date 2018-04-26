
def main():

    from stick_slip_learn import command_line
    import time
    

    print("".join(["-"]*108),"\n beginning preparing the data files\n", "".join(["-"]*108))
    time_begin = time.time()
    command_line.main() 
    time_end = time.time() 

    print("".join(["-"]*108),"\nthe time to create all files was ", time_end-time_begin, "\n", "".join(["-"]*108))


if __name__ == "__main__":
    main()
