<template>
  <div class="w-screen h-screen flex flex-col text-white cuerpo">
    <div class="menuSection"> 
      <ModelSelection />
    </div>
    <div class="chatSection"> 
      <Chat/>
    </div>
    
  </div>
</template>

<script setup>
import { ref } from "vue";
import ModelSelection from './components/ModelSelection.vue';
import Chat from './components/Chat.vue';

const messages = ref(["Welcome to the chat!", "How can I help you?"]);
const newMessage = ref("");

const sendMessage = () => {
  if (newMessage.value.trim() !== "") {
    messages.value.push(newMessage.value);
    newMessage.value = "";
  }
};
</script>

<style scoped>
.cuerpo {
  padding: 10px;
  display: flex;
  flex-direction: column;
  width: 100vw;
  height: 100vh; 
  padding-bottom:10vh;
  background-color: #f5f5f5;

}


.menuSection{
  min-height: 55px;
  width: 100%;
}

.chatSection{ 
  flex-grow: 1; 
  width: 50%;
  margin-left: 25%;
}


</style>