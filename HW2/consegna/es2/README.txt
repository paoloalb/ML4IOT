We notices that on different OS or different PCs, the way python lists the directories changes.
This means that the order of LABELS could change if it is obtained directly from the directory names.
To solve this issue, the order of LABELS is fixed to this one:

LABELS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

So during testing, keep in mind that models predictions will follow this order.
