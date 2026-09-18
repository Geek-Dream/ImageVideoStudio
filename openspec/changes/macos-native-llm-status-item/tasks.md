# macOS Native LLM Status Item Tasks

- [x] Add contract tests for status mapping and text formatting (tests are the executable requirement; implementation must make them pass).
- [x] Implement `mac_status_item.py` as an independent supervisor: create the AppKit item only while a model is running, poll status, render the menu, and handle PyObjC/import failure gracefully.
- [x] Update `start.sh` and `llm.start_llm()` to start the macOS status item idempotently, expose the persistent option-3 enable/disable switch, and stop/clean its PID with the project.
- [x] Add focused unit/shell tests and run the existing test suite; record results in this change.
- [ ] Perform macOS manual verification with browser closed; confirm loading, prompt, generation, idle, and error displays (user-side acceptance).
