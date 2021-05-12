import os  # accessing the os functions
import testing_humans_and_horses
#import testingHandRecognition
import ObjectDetection
import testingGender
import LandmarkDrowsinessDetection


# creating the title bar function

def title_bar():
    os.system('cls')

    print("\t**********************************************")
    print("\t***** Hyperface A deeplearning Multitask learning for Objection Detection"
          "\n\tHumans & Horses Detection"
          "\n\tHand Recognition"
          "\n\tLandmark Drowsiness Detection"
          "\n\tTesting Gender ******")
    print("\t**********************************************")


# creating the user main menu function

def mainMenu():
    title_bar()
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Compare Humans and Horses:")
    print("[2] HandRecoginition: ")
    print("[3] Objection Detection")
    print("[4] Gender Detection")
    print("[5] Landmark Drowsiness")
    print("[6] Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                testingHumansandHorses()
                break
            # elif choice == 2:
            #     handreco()
            #     break
            elif choice == 3:
                objectdetection()
                break
            elif choice == 4:
                gender()
                break
            elif choice == 5:
                drowsiness()
                break
            elif choice == 6:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-5")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-5\n Try Again")
    exit


# calling the camera test function from check camera.py file

def testingHumansandHorses():
    testing_humans_and_horses.testing()
    key = input("Enter any key to return main menu")
    mainMenu()


# calling the take image function form capture image.py file

# def handreco():
#     testingHandRecognition.testing()
#     key = input("Enter any key to return main menu")
#     mainMenu()


# calling the train images from train_images.py file

def objectdetection():
    ObjectDetection.detection()
    key = input("Enter any key to return main menu")
    mainMenu()


# calling the recognize_attendance from recognize.py file

def gender():
    testingGender.testing()
    key = input("Enter any key to return main menu")
    mainMenu()

def drowsiness():
    LandmarkDrowsinessDetection.Landmark()
    key = input("Enter any key to return main menu")
    mainMenu()


mainMenu()
