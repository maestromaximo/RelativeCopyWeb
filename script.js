$(document).ready(function() {
    $('#inputForm').on('submit', function(e) {
        e.preventDefault();

        var userInput = $('#userInput').val();
        $('#userInput').val('');

        // Display the user's message
        $('#chat').append('<div class="user"><b>You:</b> ' + userInput + '</div>');

        // Hide the greeting
        $('#greeting').hide();

        // Scroll the chat
        $('#chatContainer').scrollTop($('#chatContainer')[0].scrollHeight);

        // Send the user's message to the server
        $.ajax({
            url: 'http://localhost:5000/api/chat',  // Updated this line
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ 'prompt': userInput }),
            success: function(data) {
                // Display the assistant's message
                $('#chat').append('<div class="assistant"><b>Assistant:</b> ' + data.response + '</div>');

                // Scroll the chat
                $('#chatContainer').scrollTop($('#chatContainer')[0].scrollHeight);
            }
        });
    });
});
