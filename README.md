# FYP

**Code files**    

* _togglePause()_: Toggles the pause state of the game engine. If the engine is not null, it checks whether the engine is currently paused. If is it currently paused, it calls the countdown method to resume the game; otherwise, it pauses the engine and updates the UI. 

* _startResumeCountdown()_: If engine is already paused, initiate countdown and then call the resume method. Removes a CSS style class from the UI root element.
  
* _countdownAndResume()_: Displays the countdown on the screen. After the countdown, resumes the game engine by updating the UI on the JavaFX application thread.

* _handle(KeyEvent event)_: Contains a case where if Space Bar is clicked, it calls the `togglePause()` method.

* _Style.css_: Added a .pauseRoot sections with background color set to #c78aa3. This will be added when the game is paused and removed once game is resumed.
