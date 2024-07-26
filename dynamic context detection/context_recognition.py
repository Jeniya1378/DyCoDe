import os
import tkinter as tk
from tkinter import ttk, messagebox
import speech_recognition as sr
from pygame import mixer
import csv
from Model_load.model_loading import prediction

# for i in ["Dataset", "Context Detection"]:
#     if (not(os.path.exists(f"{i}/"))):
#         os.mkdir(i)


# def save_text(text_before_context, text_after_context):
#     filename = 'Dataset/output.xlsx'
#     if os.path.exists(filename):
#         wb = load_workbook(filename)
#         ws = wb.active
#         next_row = ws.max_row + 1
#     else:
#         wb = Workbook()
#         ws = wb.active
#         next_row = 1

#     # Save the detected text and context into separate columns
#     ws.cell(row=next_row, column=1, value=text_before_context)
#     ws.cell(row=next_row, column=2, value=text_after_context)
    
#     wb.save(filename)

def process_score(text_before_context, context, confidence_score):
    misclassified_sentence = {}
    if confidence_score < 0.4:
        misclassified_sentence = {'sentence': text_before_context, 'context': context, 'score': float(confidence_score)}
        # Append the misclassified sentence to the existing CSV file
        with open('misclassified_sentence.csv', 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['sentence', 'context', 'score'])
            # Check if the file is empty, if so, write header
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(misclassified_sentence)

def activityExecution(): 
    selected_option = activity.get()

    # if selected_option == 'Save to Dataset':
    #     # Get the text from the entry fields
    #     text_before_context = detectedTextEntry.get()
    #     text_after_context = contextEntry.get()
        
    #     # Save the transcription to Excel
    #     save_text(text_before_context, text_after_context) 
    
    # elif selected_option == 'Detect Context':
    # Get the text from the entry field
    text_before_context = detectedTextEntry.get()
    context, confidence_score = prediction(text_before_context)

    contextEntry.delete(0, tk.END)
    contextEntry.insert(0, context)

    scoreEntry.delete(0, tk.END)
    scoreEntry.insert(0, confidence_score)
    process_score(text_before_context, context, confidence_score)



# def saveRating():
#     filename = 'Dataset/ratings.xlsx'
#     rating = dropdown.get()
#     if os.path.exists(filename):
#         wb = load_workbook(filename)
#         ws = wb.active
#         next_row = ws.max_row + 1
#     else:
#         wb = Workbook()
#         ws = wb.active
#         next_row = 1

#     ws.cell(row=next_row, column=1, value=rating)
        
#     wb.save(filename)

def clearEntries():
    detectedTextEntry.delete(0, tk.END)
    contextEntry.delete(0, tk.END)
    scoreEntry.delete(0, tk.END)

def recordButtonClick():
    mixer.init()
    mixer.music.load('chime1.mp3')
    mixer.music.play()

    r = sr.Recognizer()
    r.pause_threshold = 0.7
    r.energy_threshold = 800
    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=7, phrase_time_limit=7)
            message = str(r.recognize_google(audio))
            text_before_context = message


            # parts = message.split("for context", 1)
            # if len(parts) == 2:
            #     text_before_context = parts[0].strip()
            #     text_after_context = parts[1].strip()
            # else:
            #     text_before_context = message.strip()
            #     text_after_context = ""



            detectedTextEntry.focus()
            detectedTextEntry.delete(0, tk.END)
            detectedTextEntry.insert(0, text_before_context)
            # contextEntry.delete(0, tk.END)
            # contextEntry.insert(0, text_after_context)
            mixer.music.load('chime2.mp3')
            mixer.music.play()

            with open("microphone-results.wav", "wb") as f:
                f.write(audio.get_wav_data())

  
            # save_text(text_before_context, text_after_context)  
            
        except sr.WaitTimeoutError:
            messagebox.showerror("Error", "Listening timed out while waiting for phrase to start")
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            messagebox.showerror("Error", f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        else:
            print("Passing")
            pass


root = tk.Tk()
root.title('Recorder')
root.iconbitmap('mic.ico')
style = tk.ttk.Style()
style.theme_use('winnative')

photo = tk.PhotoImage(file='microphone.png').subsample(35, 35)


detectedTextLabel = tk.ttk.Label(root, text='Detected Voice Command')
detectedTextLabel.grid(row=0, column=0)

detectedTextEntry = tk.ttk.Entry(root, width=50)
detectedTextEntry.grid(row=0, column=3, columnspan=2)



activityLabel = tk.ttk.Label(root, text='Activity')
activityLabel.grid(row=1, column=0)

activity = tk.StringVar(root)
activity.set('Context Recognition')

activityPopupMenu = tk.OptionMenu(root, activity, *{'Context Recognition': 'Context Recognition'})
activityPopupMenu.grid(row=1, column=2, columnspan=2)

recordButton = tk.ttk.Button(root, image=photo, command=recordButtonClick)
recordButton.grid(row=1, column=3, columnspan=2)


saveOrDetectButton = tk.ttk.Button(
    root, text='Click to identify', width=16, command=activityExecution)
saveOrDetectButton.grid(row=1, column=5)

clearButton = tk.ttk.Button(
    root, text='Clear', width=12, command=clearEntries)
clearButton.grid(row=1, column=7)

contextLabel = tk.ttk.Label(root, text='Context')
contextLabel.grid(row=2, column=0, columnspan=2)

contextEntry = tk.ttk.Entry(root, width=50)
contextEntry.grid(row=2, column=1, columnspan=4)

scoreLabel = tk.ttk.Label(root, text='Score')
scoreLabel.grid(row=2, column=5, columnspan=2)

scoreEntry = tk.ttk.Entry(root, width=50)
scoreEntry.grid(row=2, column=6, columnspan=4)

# dropdownLabel = tk.ttk.Label(root, text='Enter Rating')
# dropdownLabel.grid(row=3, column=5)

# dropdown = tk.IntVar(root)
# dropdown.set(1)
# dropdownMenu = ttk.Combobox(root, textvariable=dropdown, values=[1, 2, 3, 4, 5])
# dropdownMenu.grid(row=3, column=6)


# saveRatingButton = tk.ttk.Button(
#     root, text='Save Rating', width=12, command=saveRating)
# saveRatingButton.grid(row=3, column=7)

root.wm_attributes('-topmost', 1)
root.mainloop()
