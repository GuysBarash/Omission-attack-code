import subprocess

if __name__ == '__main__':
    print "GO"

    msg = subprocess.call(r"net use /del R: /y", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    msg = subprocess.call(r"net use R: \\10.24.8.10\tfnlab Qwerty123! /user:sdcorp\autodev.service /persistent:yes",
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
