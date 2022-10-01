import train_original
import train_custom
import test_original
import test_custom
import mAP_evaluation


def main():
    print("Yolov3 comparison solution!")
    print("Goal of this solution is to make comparison in between 2 object detection algorithms")
    print("Solution compare Yolov3 original version and Yolov3 custom with some general improvements")
    print("")
    print("What do you want to do? (Press 1-8 and enter)")
    print("1-train original algorithm")
    print("2-train custom algorithm")
    print("3-test original algorithm")
    print("4-test custom algorithm")
    print("5-evaluate original algorithm against noised images")
    print("6-evaluate custom algorithm against noised images")
    choice = input()

    if choice == '1':
        train_original.main_train_original()
    elif choice == '2':
        train_custom.main_train_custom()
    elif choice == '3':
        test_original.main_test_original()
    elif choice == '4':
        test_custom.main_test_custom()
    elif choice == '5':
        mAP_evaluation.evaluate(0, 1)
    elif choice == '6':
        mAP_evaluation.evaluate(1, 1)

if __name__ == '__main__':
    main()
