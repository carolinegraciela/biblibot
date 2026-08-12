let isInputHidden = true;

const observer = new MutationObserver(function(mutations, me) {
    const chatInputContainer = document.querySelector('div.relative.flex.flex-col.items-center.gap-2.w-full');
    
    if (chatInputContainer) {
        if (isInputHidden) {
            chatInputContainer.style.setProperty('display', 'none', 'important');
        } else {
            chatInputContainer.style.setProperty('display', 'flex', 'important');
        }
    }
});

observer.observe(document.body, { childList: true, subtree: true });

document.addEventListener('click', function(event) {
    const target = event.target.closest('button') || event.target;
    
    const buttonText = target.textContent || target.innerText || "";

    if (buttonText.includes("Tanya Jawab")) {
        isInputHidden = false; 
        
        const chatInputContainer = document.querySelector('div.relative.flex.flex-col.items-center.gap-2.w-full');
        if (chatInputContainer) {
            chatInputContainer.style.setProperty('display', 'flex', 'important');
        }
    }
    
    const isNewChatIcon = target.closest('#new-chat-button') || (target.tagName.toLowerCase() === 'svg' && target.closest('a'));
    if (isNewChatIcon) {
        isInputHidden = true; }
});