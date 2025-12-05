import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext
from tkinterdnd2 import TkinterDnD, DND_FILES
import asyncio
import threading
import re
import os
import wave
import pyaudio
import whisper
import tempfile
from pypdf import PdfReader
from google.genai import types
from agent import runner, memory_service, get_or_create_session

# Constants
USER_ID = "Noah_Haag"
SESSION_ID = "Job_Search"

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TkinterDnD_CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class AgentGUI(TkinterDnD_CTk):
    def __init__(self):
        super().__init__()

        self.title("Agent V2 Lite UI")
        self.geometry("900x700")

        # Configure Grid Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)

        # Enable Drag & Drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop_file)

        # Chat History Frame
        self.chat_frame = ctk.CTkFrame(self, corner_radius=10)
        self.chat_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)

        # Chat History Text Area
        self.chat_history = scrolledtext.ScrolledText(
            self.chat_frame, 
            state='disabled', 
            wrap=tk.WORD, 
            font=("Segoe UI", 12),
            bg="#2b2b2b",
            fg="#ffffff",
            insertbackground="white",
            selectbackground="#4a90e2",
            borderwidth=0,
            highlightthickness=0
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure Tags
        self.chat_history.tag_config("user_header", foreground="#98c379", font=("Segoe UI", 11, "bold"))
        self.chat_history.tag_config("agent_header", foreground="#61afef", font=("Segoe UI", 11, "bold"))
        self.chat_history.tag_config("system_header", foreground="#e06c75", font=("Segoe UI", 11, "bold"))
        self.chat_history.tag_config("bold", font=("Segoe UI", 12, "bold"))
        self.chat_history.tag_config("code", background="#1e1e1e", foreground="#d4d4d4", font=("Consolas", 11))
        self.chat_history.tag_config("separator", foreground="#555555")

        # Input Area Frame
        self.input_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        # Input Text
        self.user_input = ctk.CTkTextbox(
            self.input_frame, 
            height=80, 
            font=("Segoe UI", 12),
            activate_scrollbars=True
        )
        self.user_input.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        
        self.user_input.bind("<Return>", self.on_enter_pressed)
        self.user_input.bind("<Shift-Return>", lambda e: None)

        # Microphone Button
        self.mic_btn = ctk.CTkButton(
            self.input_frame, 
            text="🎤", 
            command=self.toggle_recording, 
            width=50,
            height=80,
            font=("Segoe UI", 20),
            fg_color="#3c3f41",
            hover_color="#4a90e2"
        )
        self.mic_btn.grid(row=0, column=1, sticky="ns", padx=(0, 5))
        
        # Send Button
        self.send_btn = ctk.CTkButton(
            self.input_frame, 
            text="SEND", 
            command=self.send_message, 
            width=100,
            height=80,
            font=("Segoe UI", 12, "bold")
        )
        self.send_btn.grid(row=0, column=2, sticky="ns")
        
        # Recording state
        self.is_recording = False
        self.whisper_model = None

        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_bar = ctk.CTkLabel(
            self, 
            textvariable=self.status_var, 
            anchor="w",
            font=("Segoe UI", 10),
            text_color="#abb2bf"
        )
        self.status_bar.grid(row=2, column=0, padx=15, pady=(0, 5), sticky="ew")

        # Asyncio Loop
        self.loop = asyncio.new_event_loop()
        self.agent_thread = threading.Thread(target=self.start_async_loop, daemon=True)
        self.agent_thread.start()

        # Initialize Session
        self.run_async(self.initialize_session())

    def start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_async(self, coro):
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def initialize_session(self):
        try:
            self.session = await get_or_create_session(USER_ID, SESSION_ID)
            await memory_service.add_session_to_memory(self.session)
            self.schedule_ui_update(self.update_status, "Agent Connected. Ready. (Drag & Drop files supported)")
            self.schedule_ui_update(self.append_message, "System", "Agent connected successfully.")
        except Exception as e:
            self.schedule_ui_update(self.update_status, f"Error connecting: {e}")
            self.schedule_ui_update(self.append_message, "System", f"Error: {e}")

    def drop_file(self, event):
        file_path = event.data
        # Handle curly braces if path has spaces (TkinterDnD quirk)
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
            
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            header = f"\n--- File: {filename} ---\n"
            
            try:
                # Check if it's a PDF
                if file_path.lower().endswith('.pdf'):
                    reader = PdfReader(file_path)
                    content = ""
                    for page_num, page in enumerate(reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            content += f"\n[Page {page_num}]\n{page_text}\n"
                    
                    if not content.strip():
                        content = "(PDF appears to be empty or text could not be extracted)"
                else:
                    # Try reading as text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                # Insert into input box
                self.user_input.insert("end", header + content + "\n")
                self.update_status(f"Loaded file: {filename}")
            except Exception as e:
                self.update_status(f"Error reading file: {e}")
                self.append_message("System", f"Error reading file {file_path}: {e}")

    def toggle_recording(self):
        """Toggle audio recording on/off."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.mic_btn.configure(fg_color="#e06c75", text="⏹")
        self.update_status("Recording... Click mic again to stop")
        
        # Run recording in background thread
        threading.Thread(target=self.record_audio, daemon=True).start()
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        self.mic_btn.configure(fg_color="#3c3f41", text="\ud83c\udfa4")
        self.update_status("Processing audio...")
    
    def record_audio(self):
        """Record audio from microphone."""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        
        # List available devices for debugging
        print("\nAvailable audio devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']}")
        
        # Try to use Realtek Microphone Array
        device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if 'Microphone Array' in info['name'] and 'Realtek' in info['name'] and info['maxInputChannels'] > 0:
                device_index = i
                print(f"\nUsing device {i}: {info['name']}")
                break
        
        if device_index is None:
            # Use default
            print("\nUsing default microphone")
        
        try:
            stream = p.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           input=True,
                           input_device_index=device_index,
                           frames_per_buffer=CHUNK)
        except Exception as e:
            self.schedule_ui_update(self.update_status, f"Microphone error: {e}")
            p.terminate()
            self.is_recording = False
            self.mic_btn.configure(fg_color="#3c3f41", text="\ud83c\udfa4")
            return
        
        frames = []
        duration = 0
        max_amplitude = 0
        
        while self.is_recording:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                duration += CHUNK / RATE
                
                # Check audio level
                import numpy as np
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                amplitude = np.abs(audio_chunk).max()
                max_amplitude = max(max_amplitude, amplitude)
                
                # Update status with duration and level
                level_bar = "█" * min(int(amplitude / 3000), 10)
                self.schedule_ui_update(self.update_status, f"Recording {duration:.1f}s [{level_bar}]")
            except Exception as e:
                print(f"Recording error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print(f"\nMax audio amplitude: {max_amplitude} (should be > 1000 for speech)")
        
        if len(frames) == 0:
            self.schedule_ui_update(self.update_status, "No audio recorded")
            return
        
        if max_amplitude < 100:
            self.schedule_ui_update(self.update_status, "Audio too quiet - check microphone!")
            return
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Recorded {duration:.1f}s of audio to {temp_file.name}")
        
        # Transcribe in background
        threading.Thread(target=self.transcribe_audio, args=(temp_file.name,), daemon=True).start()
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper."""
        try:
            # Load model on first use
            if self.whisper_model is None:
                self.schedule_ui_update(self.update_status, "Loading Whisper model...")
                self.whisper_model = whisper.load_model("tiny")
            
            # Load audio directly from WAV file (bypass ffmpeg)
            print(f"Transcribing {audio_file}...")
            import numpy as np
            
            # Read WAV file directly
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe using the audio array directly
            result = self.whisper_model.transcribe(audio_data,
             fp16=False,
             language="en",
             temperature=0.0,
             beam_size=5,
             initial_prompt="Job interview practice session.",
             without_timestamps=True,
             condition_on_previous_text=True, 
             no_speech_threshold=0.6,
             word_timestamps=False) 
            print(f"Whisper result: {result}")
            text = result["text"].strip()
            print(f"Extracted text: '{text}' (length: {len(text)})")
            
            # Insert into input box
            if text:
                self.schedule_ui_update(lambda: self.user_input.insert("end", text + " "))
                self.schedule_ui_update(self.update_status, "Transcription complete!")
            else:
                self.schedule_ui_update(self.update_status, "No speech detected")
        except Exception as e:
            print(f"Transcription exception: {e}")
            import traceback
            traceback.print_exc()
            self.schedule_ui_update(self.update_status, f"Transcription error: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(audio_file)
            except:
                pass

    def on_enter_pressed(self, event):
        if event.state & 0x0001: 
            return None 
        self.send_message()
        return "break"

    def send_message(self):
        text = self.user_input.get("1.0", "end").strip()
        if not text:
            return

        self.user_input.delete("1.0", "end")
        self.append_message("You", text)
        self.update_status("Agent is thinking...")
        
        self.run_async(self.process_agent_turn(text))

    async def process_agent_turn(self, text):
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=text)]
        )
        
        full_response = ""
        try:
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=user_message
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        text_parts = [p.text for p in event.content.parts if p.text]
                        if text_parts:
                            chunk = "\n".join(text_parts)
                            full_response = chunk
            
            if full_response:
                self.schedule_ui_update(self.append_message, "Agent", full_response)
                self.schedule_ui_update(self.update_status, "Ready.")
            else:
                 self.schedule_ui_update(self.append_message, "Agent", "(No text response)")
                 self.schedule_ui_update(self.update_status, "Ready.")

        except Exception as e:
            self.schedule_ui_update(self.append_message, "System", f"Error: {e}")
            self.schedule_ui_update(self.update_status, "Error.")

    def schedule_ui_update(self, func, *args):
        self.after(0, func, *args)

    def append_message(self, sender, message):
        self.chat_history.configure(state='normal')
        
        tag = "user_header" if sender == "You" else "agent_header" if sender == "Agent" else "system_header"
        self.chat_history.insert(tk.END, f"[{sender}]\n", tag)
        
        self.insert_markdown_text(message)
        
        self.chat_history.insert(tk.END, "\n" + "-"*50 + "\n\n", "separator")
        
        self.chat_history.configure(state='disabled')
        self.chat_history.see(tk.END)

    def insert_markdown_text(self, text):
        parts = re.split(r'(```[\s\S]*?```)', text)
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                code_content = part[3:-3].strip()
                self.chat_history.insert(tk.END, f"\n{code_content}\n", "code")
            else:
                self.insert_bold_text(part)

    def insert_bold_text(self, text):
        parts = re.split(r'(\*\*.*?\*\*)', text)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                content = part[2:-2]
                self.chat_history.insert(tk.END, content, "bold")
            else:
                self.chat_history.insert(tk.END, part)

    def update_status(self, text):
        self.status_var.set(text)

if __name__ == "__main__":
    app = AgentGUI()
    app.mainloop()
