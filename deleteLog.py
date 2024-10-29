import os

def deleteLog():
    path = "C:/Users/Max/AppData/Local/Temp/"

    cpt = 0
    # Delete every folder starting with "SB3"
    for folder in os.listdir(path):
        if folder.startswith("SB3"):
            os.rmdir(path + folder)
            cpt += 1

            if cpt % 10_000 == 0:
                print(f"{cpt} folders deleted")

if __name__ == "__main__":
    deleteLog()